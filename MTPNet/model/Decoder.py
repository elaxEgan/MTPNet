import torch.nn as nn
import torch
from .transformer_block import Block2
from .Transformer import saliency_token_inference, edge_token_inference, token_TransformerEncoder


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_trans, self).__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                                     mlp_ratio=mlp_ratio)
        self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)
        self.edge_token_pre = edge_token_inference(dim=embed_dim, num_heads=1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm1_c = nn.LayerNorm(embed_dim)
        self.mlp1_c = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm2_c = nn.LayerNorm(embed_dim)
        self.mlp2_c = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, fea, saliency_tokens, edge_tokens, task_prompt, num):
        B, _, _ = fea.shape
        # fea [B, H*W, 64]
        # project to 384 dim
        fea = self.mlp(self.norm(fea))
        # fea [B, H*W, 384]

        fea = torch.cat((task_prompt, fea), dim=1)
        fea = torch.cat((saliency_tokens, fea), dim=1)
        fea = torch.cat((fea, edge_tokens), dim=1)
        # [B, 1 + H*W + 1, 384]

        fea, saliency_tokens, edge_tokens, fea_tmp, fea_s, task_prompt = self.encoderlayer(fea, num)

        # reproject back to 64 dim
        saliency_tokens_tmp = self.mlp1(self.norm1(saliency_tokens))
        edge_tokens_tmp = self.mlp1_c(self.norm1_c(edge_tokens))

        saliency_fea = self.saliency_token_pre(fea_s, num)
        # saliency_fea [B, H*W, 384]
        edge_fea = self.edge_token_pre(fea_s, num)
        # edge_fea [B, H*W, 384]

        # reproject back to 64 dim
        saliency_fea = self.mlp2(self.norm2(saliency_fea))
        edge_fea = self.mlp2_c(self.norm2_c(edge_fea))

        return fea, saliency_tokens, edge_tokens, fea_tmp, saliency_tokens_tmp, edge_tokens_tmp, saliency_fea, edge_fea, task_prompt


class decoder_module(nn.Module):
    def __init__(self, dim=384, token_dim=96, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                 fuse=True):
        super(decoder_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio, img_size // ratio), kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim * 2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            # self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, dec_fea, enc_fea=None):

        B, _, C = dec_fea.shape
        if C == 384:
            # from 384 to 64
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        # [B, HW, C]

        if self.fuse:
            # [2] fuse encoder fea and decoder fea
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
            # dec_fea = self.att(dec_fea)

        return dec_fea


class Task_Interaction(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64):
        super(Task_Interaction, self).__init__()
        self.MHSA = Block2(dim=embed_dim, num_heads=6)

    def forward(self, feature1, feature2):
        cat = torch.cat((feature1, feature2), dim=1)
        out = self.MHSA(cat)
        feature1_out, feature2_out = torch.split(out, feature1.size(1), dim=1)

        return feature1_out, feature2_out


class Task_split(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super(Task_split, self).__init__()
        self.num_heads = num_heads
        dim = embed_dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(embed_dim)
        self.norm_e = nn.LayerNorm(embed_dim)
        self.norm_ts = nn.LayerNorm(dim)
        self.norm_te = nn.LayerNorm(dim)

        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.v1 = nn.Linear(dim, dim, bias=qkv_bias)

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.v2 = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature1, feature2, token1, token2):
        B, N, C = feature1.shape

        feature1 = self.norm(feature1)
        feature2 = self.norm_e(feature2)
        token1 = self.norm_ts(token1)
        token2 = self.norm_te(token2)

        q1 = self.q1(feature1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k1 = self.k1(token1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v1 = self.v1(token1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q2 = self.q2(feature2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k2 = self.k2(token2).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v2 = self.v2(token2).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale

        attn1 = self.sigmoid(attn1)
        attn1 = self.attn_drop(attn1)

        infer_fea1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
        infer_fea1 = self.proj1(infer_fea1)
        infer_fea1 = self.proj_drop(infer_fea1)

        infer_fea1 = infer_fea1 + feature1

        attn2 = self.sigmoid(attn2)
        attn2 = self.attn_drop(attn2)

        infer_fea2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C)
        infer_fea2 = self.proj2(infer_fea2)
        infer_fea2 = self.proj_drop(infer_fea2)

        infer_fea2 = infer_fea2 + feature2

        return infer_fea1, infer_fea2


class Task_Interaction2(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64):
        super(Task_Interaction2, self).__init__()
        self.MHSA = Block2(dim=token_dim, num_heads=8)

    def forward(self, feature1, feature2):
        cat = torch.cat((feature1, feature2), dim=1)
        out = self.MHSA(cat)
        feature1_out, feature2_out = torch.split(out, feature1.size(1), dim=1)

        return feature1_out, feature2_out


class Task_split2(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super(Task_split2, self).__init__()
        self.num_heads = num_heads
        dim = token_dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.norm_e = nn.LayerNorm(dim)
        self.norm_ts = nn.LayerNorm(dim)
        self.norm_te = nn.LayerNorm(dim)

        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.v1 = nn.Linear(dim, dim, bias=qkv_bias)

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.v2 = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.projt1 = nn.Linear(embed_dim, dim)
        self.projt2 = nn.Linear(embed_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature1, feature2, token1, token2):
        B, N, C = feature1.shape

        feature1 = self.norm(feature1)
        feature2 = self.norm_e(feature2)
        token1 = self.projt1(token1)
        token2 = self.projt2(token2)
        token1 = self.norm_ts(token1)
        token2 = self.norm_te(token2)

        q1 = self.q1(feature1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k1 = self.k1(token1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v1 = self.v1(token1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q2 = self.q2(feature2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k2 = self.k2(token2).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v2 = self.v2(token2).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale

        attn1 = self.sigmoid(attn1)
        attn1 = self.attn_drop(attn1)

        infer_fea1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
        infer_fea1 = self.proj1(infer_fea1)
        infer_fea1 = self.proj_drop(infer_fea1)

        infer_fea1 = infer_fea1 + feature1

        attn2 = self.sigmoid(attn2)
        attn2 = self.attn_drop(attn2)

        infer_fea2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C)
        infer_fea2 = self.proj2(infer_fea2)
        infer_fea2 = self.proj_drop(infer_fea2)

        infer_fea2 = infer_fea2 + feature2

        return infer_fea1, infer_fea2


class Task_Interaction_token(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64):
        super(Task_Interaction_token, self).__init__()
        self.MHSA = Block2(dim=token_dim, num_heads=8)
        self.norm1_s = nn.LayerNorm(token_dim)
        self.mlp1_s = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.norm1_e = nn.LayerNorm(token_dim)
        self.mlp1_e = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
    def forward(self, token1, token2):
        cat = torch.cat((token1, token2), dim=1)
        out = self.MHSA(cat)
        token1_out, token2_out = torch.split(out, token1.size(1), dim=1)
        token1_out = self.mlp1_s(self.norm1_s(token1_out))
        token2_out = self.mlp1_e(self.norm1_e(token2_out))

        return token1_out, token2_out


class Decoder(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, depth=2, img_size=224):

        super(Decoder, self).__init__()

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm2_e = nn.LayerNorm(embed_dim)
        self.mlp2_e = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm_e = nn.LayerNorm(embed_dim)
        self.mlp_e = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm_fgall = nn.LayerNorm(embed_dim * 2)
        self.mlp_fgall = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.img_size = img_size
        self.token_dim = token_dim
        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=4,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)

        self.decoder3 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1,
                                       kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)
        self.decoder3_s = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1,
                                         kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)
        self.decoder3_c = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1,
                                         kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)

        # token based multi-task predictions
        self.token_pre_1_8 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)
        self.token_pre_1_4 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)

        self.pre_1_16 = nn.Linear(token_dim, 1)
        self.pre_1_8 = nn.Linear(token_dim, 1)
        self.pre_1_4 = nn.Linear(token_dim, 1)
        self.pre_1_1 = nn.Linear(token_dim, 1)
        # predict edge maps
        self.pre_1_16_e = nn.Linear(token_dim, 1)
        self.pre_1_8_e = nn.Linear(token_dim, 1)
        self.pre_1_4_e = nn.Linear(token_dim, 1)
        self.pre_1_1_e = nn.Linear(token_dim, 1)

        self.enc_dim_8 = nn.Linear(embed_dim // 2, token_dim)
        self.enc_dim_4 = nn.Linear(embed_dim // 4, token_dim)


        self.Task_Interaction1 = Task_Interaction()
        self.Task_split1 = Task_split()

        self.Task_Interaction2 = Task_Interaction2()
        self.Task_split2 = Task_split2()

        self.Task_Interaction3 = Task_Interaction2()
        self.Task_split3 = Task_split2()

        self.Task_Interaction4 = Task_Interaction2()
        self.Task_split4 = Task_split2()

        self.Task_Interaction_token4 = Task_Interaction_token()
        self.Task_Interaction_token3 = Task_Interaction_token()
        self.Task_Interaction_token2 = Task_Interaction_token()
        self.Task_Interaction_token1 = Task_Interaction_token()

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fea_1_16, saliency_tokens, edge_tokens, fea_16, saliency_tokens_16, edge_tokens_16,
                saliency_fea_1_16, edge_fea_1_16, rgb_fea_1_8, rgb_fea_1_4, task_prompt, num):

        B, _, _, = fea_1_16.size()
        dim_d = torch.sqrt(torch.tensor(self.token_dim).to(torch.double))

        saliency_tokens_16, edge_tokens_16 = self.Task_Interaction_token4(saliency_tokens_16, edge_tokens_16)
        mask_1_16 = (fea_16 @ (saliency_tokens_16.permute(0, 2, 1))) / dim_d
        mask_1_16 = mask_1_16.reshape(B, 1, self.img_size // 16, self.img_size // 16)


        edg_1_16 = (fea_16 @ edge_tokens_16.permute(0, 2, 1)) / dim_d
        edg_1_16 = edg_1_16.reshape(B, 1, self.img_size // 16, self.img_size // 16)


        #fea1, fea2 = self.Task_Interaction1(saliency_fea_1_16, edge_fea_1_16)
        #fea1, fea2 = self.Task_split1(fea1, fea2, saliency_tokens, edge_tokens)
        saliency_fea_1_16 = self.mlp(self.norm(saliency_fea_1_16))
        # saliency_fea_1_16 [B, 14*14, 64]
        mask_1_16_s = self.pre_1_16(saliency_fea_1_16)
        mask_1_16_s = mask_1_16_s.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        edge_fea_1_16 = self.mlp_e(self.norm_e(edge_fea_1_16))
        # edge_fea_1_16 [B, 14*14, 64]
        edg_1_16_s = self.pre_1_16_e(edge_fea_1_16)
        edg_1_16_s = edg_1_16_s.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        # 1/16 -> 1/8
        rgb_fea_1_8 = self.enc_dim_8(rgb_fea_1_8)
        fea_1_8 = self.decoder1(fea_1_16, rgb_fea_1_8)

        # token prediction
        fea_1_8, saliency_tokens, edge_tokens, fea_8, saliency_tokens_8, edge_tokens_8, saliency_fea_1_8, edge_fea_1_8, task_prompt = self.token_pre_1_8(
            fea_1_8, saliency_tokens, edge_tokens, task_prompt, num)

        saliency_tokens_8, edge_tokens_8 = self.Task_Interaction_token3(saliency_tokens_8, edge_tokens_8)
        mask_1_8 = (fea_8 @ (saliency_tokens_8.permute(0, 2, 1))) / dim_d
        mask_1_8 = mask_1_8.reshape(B, 1, self.img_size // 8, self.img_size // 8)

        edg_1_8 = (fea_8 @ edge_tokens_8.permute(0, 2, 1)) / dim_d
        edg_1_8 = edg_1_8.reshape(B, 1, self.img_size // 8, self.img_size // 8)

        mask_1_8_s = self.pre_1_8(saliency_fea_1_8)
        mask_1_8_s = mask_1_8_s.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)

        edg_1_8_s = self.pre_1_8_e(edge_fea_1_8)
        edg_1_8_s = edg_1_8_s.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)

        # 1/8 -> 1/4
        rgb_fea_1_4 = self.enc_dim_4(rgb_fea_1_4)
        fea_1_4 = self.decoder2(fea_1_8, rgb_fea_1_4)

        # token prediction
        fea_1_4, saliency_tokens, edge_tokens, fea_4, saliency_tokens_4, edge_tokens_4, saliency_fea_1_4, edge_fea_1_4, task_prompt = self.token_pre_1_4(
            fea_1_4, saliency_tokens, edge_tokens, task_prompt, num)

        saliency_tokens_4, edge_tokens_4 = self.Task_Interaction_token2(saliency_tokens_4, edge_tokens_4)
        mask_1_4 = (fea_4 @ saliency_tokens_4.permute(0, 2, 1)) / dim_d
        mask_1_4 = mask_1_4.reshape(B, 1, self.img_size // 4, self.img_size // 4)

        edg_1_4 = (fea_4 @ edge_tokens_4.permute(0, 2, 1)) / dim_d
        edg_1_4 = edg_1_4.reshape(B, 1, self.img_size // 4, self.img_size // 4)

        mask_1_4_s = self.pre_1_4(saliency_fea_1_4)
        mask_1_4_s = mask_1_4_s.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)
        edg_1_4_s = self.pre_1_4_e(edge_fea_1_4)
        edg_1_4_s = edg_1_4_s.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)

        # 1/4 -> 1
        fea_1_1 = self.decoder3(fea_1_4)
        saliency_fea_1_1 = self.decoder3_s(saliency_fea_1_4)
        edge_fea_1_1 = self.decoder3_c(edge_fea_1_4)

        saliency_tokens_1 = self.mlp2(self.norm2(saliency_tokens))
        edge_tokens_1 = self.mlp2_e(self.norm2_e(edge_tokens))

        mask_1_1 = (fea_1_1 @ saliency_tokens_1.permute(0, 2, 1)) / dim_d
        mask_1_1 = mask_1_1.reshape(B, 1, self.img_size, self.img_size)

        edg_1_1 = (fea_1_1 @ edge_tokens_1.permute(0, 2, 1)) / dim_d
        edg_1_1 = edg_1_1.reshape(B, 1, self.img_size, self.img_size)

        mask_1_1_s = self.pre_1_1(saliency_fea_1_1)
        mask_1_1_s = mask_1_1_s.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)
        edg_1_1_s = self.pre_1_1_e(edge_fea_1_1)
        edg_1_1_s = edg_1_1_s.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)

        return [mask_1_16, mask_1_8, mask_1_4, mask_1_1], [edg_1_16, edg_1_8, edg_1_4, edg_1_1], [mask_1_16_s,
                                                                                                  mask_1_8_s,
                                                                                                  mask_1_4_s,
                                                                                                  mask_1_1_s], [
            edg_1_16_s, edg_1_8_s, edg_1_4_s, edg_1_1_s], edg_1_4