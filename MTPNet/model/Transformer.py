import torch
from torch import nn
from .transformer_block import Block
from timm.models.layers import trunc_normal_
from matplotlib import pyplot as plt



class TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.rgb_norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, rgb_fea):

        for block in self.blocks:
            rgb_fea = block(rgb_fea)

        rgb_fea = self.rgb_norm(rgb_fea)

        return rgb_fea


class token_TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, token_dim=64, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(token_TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.mlp3 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, fea, num):
        for block in self.blocks:
            fea = block(fea)

        saliency_tokens = fea[:, 0, :].unsqueeze(1)
        task_prompt = fea[:, 1:(num + 1), :]
        edge_tokens = fea[:, -1, :].unsqueeze(1)
        fea_output = fea[:, (num + 1):-1, :]
        fea_tmp = self.mlp3(self.norm(fea_output))  # 384->64

        return fea_output, saliency_tokens, edge_tokens, fea_tmp, fea, task_prompt


class Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(Transformer, self).__init__()

        self.encoderlayer = TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                               mlp_ratio=mlp_ratio)

    def forward(self, rgb_fea):
        rgb_memory = self.encoderlayer(rgb_fea)

        return rgb_memory


class saliency_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea, num):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, 0, :].unsqueeze(1), x[:, (num + 1):-1, :]


        q = self.q(F_s).reshape(B, N - num - 2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N - num - 2, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, (num + 1):-1, :]
        return infer_fea


class edge_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea, num):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, -1, :].unsqueeze(1), x[:, (num + 1):-1, :]


        q = self.q(F_s).reshape(B, N - num - 2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # attn = attn.softmax(dim=-1)
        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N - num - 2, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, (num + 1):-1, :]
        return infer_fea


class token_Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., in_dim=64):
        super(token_Transformer, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_s = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.saliency_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.edge_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.task_promptSOD_token = nn.Parameter(torch.randn(1, 10, embed_dim))
        self.task_promptSED_token = nn.Parameter(torch.randn(1, 10, embed_dim))

        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                                     mlp_ratio=mlp_ratio, token_dim=in_dim)
        self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)
        self.edge_token_pre = edge_token_inference(dim=embed_dim, num_heads=1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, in_dim),
        )
        self.norm1_c = nn.LayerNorm(embed_dim)
        self.mlp1_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, in_dim),
        )

    def forward(self, rgb_fea, num):
        B, _, _ = rgb_fea.shape
        fea_1_16 = self.mlp_s(self.norm(rgb_fea))

        saliency_tokens = self.saliency_token.expand(B, -1, -1)
        task_promptSOD1 = self.task_promptSOD_token.expand(B // 2, -1, -1)
        task_promptSED1 = self.task_promptSED_token.expand(B // 2, -1, -1)

        task_prompt1 = torch.cat((task_promptSOD1, task_promptSED1), dim=0)

        fea_1_16 = torch.cat((task_prompt1, fea_1_16), dim=1)
        # sal task fea edg
        fea_1_16 = torch.cat((saliency_tokens, fea_1_16), dim=1)

        edge_tokens = self.edge_token.expand(B, -1, -1)
        fea_1_16 = torch.cat((fea_1_16, edge_tokens), dim=1)

        fea_1_16, saliency_tokens, contour_tokens, fea_16, fea_1_16_s, task_prompt = self.encoderlayer(fea_1_16, num)

        saliency_fea_1_16 = self.saliency_token_pre(fea_1_16_s, num)
        edge_fea_1_16 = self.edge_token_pre(fea_1_16_s, num)

        saliency_tokens_tmp = self.mlp1(self.norm1(saliency_tokens))
        edge_tokens_tmp = self.mlp1_c(self.norm1_c(edge_tokens))

        task_prompt_total = [self.task_promptSOD_token.clone(), self.task_promptSED_token.clone()]

        return fea_1_16, saliency_tokens, contour_tokens, fea_16, saliency_tokens_tmp, edge_tokens_tmp, saliency_fea_1_16, edge_fea_1_16, task_prompt, task_prompt_total

