# -*-ding:utf-8-*-
from torch import nn
import torch
import torch.nn.functional as F
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder, decoder_module
from .swin_transformer import swin_transformer



class MTPNet(nn.Module):
    def __init__(self, args, channel=64):
        super(MTPNet, self).__init__()

        self.args = args

        self.sigmoid = nn.Sigmoid()

        # task prompt for Salient and Edge
        self.task_promptS_1 = nn.Parameter(torch.randn(1, args.task_num[0], args.encoder_dim[0]))
        self.task_promptS_2 = nn.Parameter(torch.randn(1, args.task_num[1], args.encoder_dim[1]))
        self.task_promptS_3 = nn.Parameter(torch.randn(1, args.task_num[2], args.encoder_dim[2]))
        self.task_promptS_4 = nn.Parameter(torch.randn(1, args.task_num[3], args.encoder_dim[3]))

        self.task_promptE_1 = nn.Parameter(torch.randn(1, args.task_num[0], args.encoder_dim[0]))
        self.task_promptE_2 = nn.Parameter(torch.randn(1, args.task_num[1], args.encoder_dim[1]))
        self.task_promptE_3 = nn.Parameter(torch.randn(1, args.task_num[2], args.encoder_dim[2]))
        self.task_promptE_4 = nn.Parameter(torch.randn(1, args.task_num[3], args.encoder_dim[3]))

        self.num_task = args.task_num
        self.bs = args.batchsize

        self.rgb_backbone = swin_transformer(pretrained=True, args=self.args)


        self.mlp32 = nn.Sequential(
            nn.Linear(args.encoder_dim[3], args.encoder_dim[2]),
            nn.GELU(),
            nn.Linear(args.encoder_dim[2], args.encoder_dim[2]), )

        self.mlp16 = nn.Sequential(
            nn.Linear(args.encoder_dim[2], args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.dim), )

        self.mlp8 = nn.Sequential(
            nn.Linear(args.encoder_dim[1], args.encoder_dim[1]),
            nn.GELU(),
            nn.Linear(args.encoder_dim[1], args.encoder_dim[1]), )

        self.mlp4 = nn.Sequential(
            nn.Linear(args.encoder_dim[0], args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.dim), )

        self.norm1 = nn.LayerNorm(args.dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(args.dim, args.embed_dim),
            nn.GELU(),
            nn.Linear(args.embed_dim, args.embed_dim),
        )

        self.fuse_32_16 = decoder_module(dim=args.embed_dim, token_dim=args.dim, img_size=args.trainsize, ratio=16,
                                         kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)

        self.linearS1 = nn.Linear(args.encoder_dim[0], args.embed_dim)
        self.linearS2 = nn.Linear(args.encoder_dim[1], args.embed_dim)
        self.linearS4 = nn.Linear(args.encoder_dim[3], args.embed_dim)

        self.linearE1 = nn.Linear(args.encoder_dim[0], args.embed_dim)
        self.linearE2 = nn.Linear(args.encoder_dim[1], args.embed_dim)
        self.linearE4 = nn.Linear(args.encoder_dim[3], args.embed_dim)

        self.linearR_low = nn.Linear(args.embed_dim * 4, args.embed_dim)
        self.linearE_low = nn.Linear(args.embed_dim * 4, args.embed_dim)
        self.linearS_low = nn.Linear(args.embed_dim * 4, args.embed_dim)
        self.linearE_low = nn.Linear(args.embed_dim * 4, args.embed_dim)

        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.trainsize)
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        self.num_deco = 10


    def forward(self, x):

        task_prompt1 = torch.cat(
            (self.task_promptS_1.repeat(self.bs // 2, 1, 1), self.task_promptE_1.repeat(self.bs // 2, 1, 1)), dim=0)
        task_prompt2 = torch.cat(
            (self.task_promptS_2.repeat(self.bs // 2, 1, 1), self.task_promptE_2.repeat(self.bs // 2, 1, 1)), dim=0)
        task_prompt3 = torch.cat(
            (self.task_promptS_3.repeat(self.bs // 2, 1, 1), self.task_promptE_3.repeat(self.bs // 2, 1, 1)), dim=0)
        task_prompt4 = torch.cat(
            (self.task_promptS_4.repeat(self.bs // 2, 1, 1), self.task_promptE_4.repeat(self.bs // 2, 1, 1)), dim=0)

        task_prompt = [task_prompt1, task_prompt2, task_prompt3, task_prompt4]

        rgb_fea_1_4, rgb_fea_1_8, rgb_fea_1_16, rgb_fea_1_32 = self.rgb_backbone(x, task_prompt, self.num_task)


        rgb_fea_1_32 = self.mlp32(rgb_fea_1_32)
        rgb_fea_1_16 = self.mlp16(rgb_fea_1_16)
        rgb_fea_1_16 = self.fuse_32_16(rgb_fea_1_32, rgb_fea_1_16)
        rgb_fea_1_16 = self.mlp1(self.norm1(rgb_fea_1_16))
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)

        fea_1_16, saliency_tokens, edge_tokens, fea_16, saliency_tokens_tmp, edge_tokens_tmp, saliency_fea_1_16, edge_fea_1_16, task_prompt, task_prompt_total1 = self.token_trans(
            rgb_fea_1_16, self.num_deco)
        outputs_saliency, outputs_edg, outputs_saliency_s, outputs_edg_s, fea_1_1 = self.decoder(fea_1_16,
                                                                                                saliency_tokens,
                                                                                                edge_tokens, fea_16,
                                                                                                saliency_tokens_tmp,
                                                                                                edge_tokens_tmp,
                                                                                                saliency_fea_1_16,
                                                                                                edge_fea_1_16,
                                                                                                rgb_fea_1_8,
                                                                                                rgb_fea_1_4,
                                                                                                task_prompt,
                                                                                                self.num_deco)


        task_promptS = self.linearS_low(torch.cat([prompt for prompt in
                                                   [self.linearS1(self.task_promptS_1),
                                                    self.linearS2(self.task_promptS_2),
                                                    torch.mean(self.task_promptS_3, dim=1, keepdim=True),
                                                    torch.mean(self.linearS4(self.task_promptS_4), dim=1,
                                                               keepdim=True)]], dim=2))
        task_promptE = self.linearE_low(torch.cat([prompt for prompt in
                                                   [self.linearE1(self.task_promptE_1),
                                                    self.linearE2(self.task_promptE_2),
                                                    torch.mean(self.task_promptE_3, dim=1, keepdim=True),
                                                    torch.mean(self.linearE4(self.task_promptE_4), dim=1,
                                                               keepdim=True)]], dim=2))
        task_prompt_total2 = [task_promptS, task_promptE]
        all_dict = {}
        all_dict['fea_1_1'] = torch.sigmoid(fea_1_1)


        return task_prompt_total2, task_prompt_total1, outputs_saliency, outputs_edg, outputs_saliency_s, outputs_edg_s, all_dict
