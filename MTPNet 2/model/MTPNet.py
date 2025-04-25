# -*-ding:utf-8-*-
from torch import nn
import torch
from .Transformer import token_Transformer
from .Decoder import Decoder, decoder_module
from .swin_transformer import swin_transformer



class MTPNet(nn.Module):
    def __init__(self, args, channel=64):
        super(MTPNet, self).__init__()

        self.args = args


        # task prompt for Salient and Edge
        self.task_promptS_1 = nn.Parameter(torch.randn(args.batchsize, args.task_num[0], args.encoder_dim[0]))
        self.task_promptS_2 = nn.Parameter(torch.randn(args.batchsize, args.task_num[1], args.encoder_dim[1]))
        self.task_promptS_3 = nn.Parameter(torch.randn(args.batchsize, args.task_num[2], args.encoder_dim[2]))
        self.task_promptS_4 = nn.Parameter(torch.randn(args.batchsize, args.task_num[3], args.encoder_dim[3]))

        self.task_promptE_1 = nn.Parameter(torch.randn(args.batchsize, args.task_num[0], args.encoder_dim[0]))
        self.task_promptE_2 = nn.Parameter(torch.randn(args.batchsize, args.task_num[1], args.encoder_dim[1]))
        self.task_promptE_3 = nn.Parameter(torch.randn(args.batchsize, args.task_num[2], args.encoder_dim[2]))
        self.task_promptE_4 = nn.Parameter(torch.randn(args.batchsize, args.task_num[3], args.encoder_dim[3]))

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


        self.norm1 = nn.LayerNorm(args.dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(args.dim, args.embed_dim),
            nn.GELU(),
            nn.Linear(args.embed_dim, args.embed_dim),
        )

        self.fuse_32_16 = decoder_module(dim=args.embed_dim, token_dim=args.dim, img_size=args.trainsize, ratio=16,
                                         kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)


        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.trainsize)
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        self.num_deco = 10
        self.num1 = args.task_num[0]
        self.num2 = args.task_num[1]
        self.num3 = args.task_num[2]
        self.num4 = args.task_num[3]

        self.linearp1 = nn.Linear(args.encoder_dim[0], args.embed_dim)
        self.linearp2 = nn.Linear(args.encoder_dim[1], args.embed_dim)
        self.linearp3 = nn.Linear(args.encoder_dim[2], args.embed_dim)
        self.linearp4 = nn.Linear(args.encoder_dim[3], args.embed_dim)


    def forward(self, x):

        task_prompt1 = torch.cat(
            (self.task_promptS_1, self.task_promptE_1), dim=1)
        task_prompt2 = torch.cat(
            (self.task_promptS_2, self.task_promptE_2), dim=1)
        task_prompt3 = torch.cat(
            (self.task_promptS_3, self.task_promptE_3), dim=1)
        task_prompt4 = torch.cat(
            (self.task_promptS_4, self.task_promptE_4), dim=1)

        task_prompt = [task_prompt1, task_prompt2, task_prompt3, task_prompt4]
        num = self.num1 + self.num2 + self.num3 + self.num4
        rgb_fea_1_4, rgb_fea_1_8, rgb_fea_1_16, rgb_fea_1_32 = self.rgb_backbone(x, task_prompt, self.num_task)

        task_prompt1_d = self.linearp1(torch.cat(
            (self.task_promptS_1, self.task_promptE_1), dim=1))
        task_prompt2_d = self.linearp2(torch.cat(
            (self.task_promptS_2, self.task_promptE_2), dim=1))
        task_prompt3_d = self.linearp3(torch.cat(
            (self.task_promptS_3, self.task_promptE_3), dim=1))
        task_prompt4_d = self.linearp4(torch.cat(
            (self.task_promptS_4, self.task_promptE_4), dim=1))

        task_prompt = torch.cat((task_prompt1_d, task_prompt2_d, task_prompt3_d, task_prompt4_d), dim=1)

        rgb_fea_1_32 = self.mlp32(rgb_fea_1_32)
        rgb_fea_1_16 = self.mlp16(rgb_fea_1_16)
        rgb_fea_1_16 = self.fuse_32_16(rgb_fea_1_32, rgb_fea_1_16)
        rgb_fea_1_16 = self.mlp1(self.norm1(rgb_fea_1_16))

        fea_1_16, saliency_tokens, edge_tokens, fea_16, saliency_tokens_tmp, edge_tokens_tmp, saliency_fea_1_16, edge_fea_1_16, task_prompt = self.token_trans(
            rgb_fea_1_16, num, task_prompt)
        outputs_saliency, outputs_edg, outputs_saliency_s, outputs_edg_s, task_prompt, fea_1_1 = self.decoder(fea_1_16,
                                                                                                saliency_tokens,
                                                                                                edge_tokens, fea_16,
                                                                                                saliency_tokens_tmp,
                                                                                                edge_tokens_tmp,
                                                                                                saliency_fea_1_16,
                                                                                                edge_fea_1_16,
                                                                                                     rgb_fea_1_8,
                                                                                                rgb_fea_1_4,
                                                                                                task_prompt,
                                                                                                num,
                                                                                               )

        all_dict = {}
        all_dict['fea_1_1'] = torch.sigmoid(fea_1_1)


        return outputs_saliency, outputs_edg, outputs_saliency_s, outputs_edg_s, task_prompt, all_dict
