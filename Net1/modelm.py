import torch
import torch.nn as nn
import math
from typing import Optional, Union, Sequence
from mmcv.cnn import ConvModule
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum
from torch.nn import functional as F
from Net1.u_Netblock import SAB, CAB, PAB, conv, SAM, conv3x3, conv_down
import torch._utils
from Net1.psacc import Block
from Net1.wavelet_block import LCA
from Net1.ISF import *
## U-Net
bn = 2  # block number-1

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, block):
        super(Encoder, self).__init__()
        if block == 'CAB':
            self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
        elif block == 'PAB':
            self.encoder_level1 = [PAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level2 = [PAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level3 = [PAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
        elif block == 'SAB':
            self.encoder_level1 = [SAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level2 = [SAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level3 = [SAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, x):
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, block):
        super(Decoder, self).__init__()
        if block == 'CAB':
            self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
        elif block == 'PAB':
            self.decoder_level1 = [PAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level2 = [PAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level3 = [PAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
        elif block == 'SAB':
            self.decoder_level1 = [SAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level2 = [SAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level3 = [SAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)
        if block == 'CAB':
            self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
            self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        if block == 'PAB':
            self.skip_attn1 = PAB(n_feat, kernel_size, reduction, bias=bias, act=act)
            self.skip_attn2 = PAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        if block == 'SAB':
            self.skip_attn1 = SAB(n_feat, kernel_size, reduction, bias=bias, act=act)
            self.skip_attn2 = SAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return [dec1, dec2, dec3]
 
    

##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        # 上采样
        x = self.up(x)

        # 打印输出形状，调试用
        # print(f"x shape: {x.shape}")
        # print(f"y shape: {y.shape}")
        
        # 调整 y 的大小使其与 x 一致
        if x.shape != y.shape:
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)  # 使用bilinear插值调整尺寸

        # 打印调整后的y形状
        # print(f"Adjusted y shape: {y.shape}")
        
        # 加法操作
        x = x + y
        return x

##########################################################################
# Mixed Residual Module
class Mix(nn.Module):
    def __init__(self, m=1):
        super(Mix, self).__init__()
        w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2, feat3):
        factor = self.mix_block(self.w)
        other = (1 - factor)/2
        output = fea1 * other.expand_as(fea1) + fea2 * factor.expand_as(fea2) + feat3 * other.expand_as(feat3)
        return output, factor

################################################################################3###
class Multi_Scale_Feature_Extract_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.Initial = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=4, stride=1, dilation=4),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_4 = nn.Sequential(
            nn.Conv2d(16 * 3, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.Initial(x)
        x1 = self.dilatation_conv_1(x)
        x2 = self.dilatation_conv_2(x)
        x3 = self.dilatation_conv_3(x)
        concatenation = torch.cat([x1, x2, x3], dim=1)
        x4 = self.dilatation_conv_4(concatenation)
        x = x4 + residual
        x = self.relu(x)
        return x



####################################################################



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.sigmoid(x)

class Ca(nn.Module):  # channel attention
    def __init__(self, in_planes, ratio=16):
        super(Ca, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        adjusted_ratio = max(1, in_planes // ratio)  # 确保最小值为1
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, adjusted_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(adjusted_ratio, in_planes, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attn = self.softmax(out)
        return attn


class CrossAttention_MP(nn.Module):
    def __init__(
            self,
            dim,
            qkv_bias=False,
            qk_scale=1 / math.sqrt(64),
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.scale = qk_scale
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        B1, C1, H1, W1 = x.shape
        x1 = x.permute(0, 2, 3, 1)
        qkv1 = self.qkv(x1)
        qkv1 = qkv1.reshape(B1, 3, C1, H1, W1).permute(1, 0, 2, 3, 4)
        q1, k1, v1 = (
            qkv1[0],
            qkv1[1],
            qkv1[2],
        )

        B2, C2, H2, W2 = y.shape
        y1 = y.permute(0, 2, 3, 1)
        qkv2 = self.qkv(y1)
        qkv2 = qkv2.reshape(B2, 3, C2, H2, W2).permute(1, 0, 2, 3, 4)
        q2, k2, v2 = (
            qkv2[0],
            qkv2[1],
            qkv2[2],
        )

        attn = torch.matmul(q2, k1.transpose(2, 3)) * self.scale

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        F = torch.matmul(attn, v1)
        return F
########################################################    
class MultiKerSize(nn.Module):
    def __init__(self, in_ch):
        super(MultiKerSize, self).__init__()
        self.sa = SpatialAttention()
        self.ca = Ca(in_ch)
        self.cross = CrossAttention_MP(dim=in_ch)

        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channels=in_ch * 2, out_channels=in_ch, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_ch),
            nn.Tanh(),
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1),  # 修改为in_ch
            nn.BatchNorm2d(in_ch),
            nn.Tanh(),
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(in_ch),
            nn.Tanh(),
        )

    def forward(self, x):


            # 3x3 branch
            fe_3x3_a = self.branch3x3(x)

            # Spatial attention and channel attention
            fe_3x3_a_sa = fe_3x3_a * self.sa(fe_3x3_a)
            fe_3x3_a_ca = fe_3x3_a * self.ca(fe_3x3_a)
            # Merging features
            fe_3x3_a_mer = self.convlayer(torch.cat((fe_3x3_a_ca, fe_3x3_a_sa), dim=1))

            # 5x5 branch
            fe_5x5_a = self.branch5x5(x)

            # Spatial and Channel Attention
            fe_5x5_a_sa = fe_5x5_a * self.sa(fe_5x5_a)
            fe_5x5_a_ca = fe_5x5_a * self.ca(fe_5x5_a)

            # Merging features
            fe_5x5_a_mer = self.convlayer(torch.cat((fe_5x5_a_sa, fe_5x5_a_ca), dim=1))

            # Cross Attention
            fe_3x3_a = self.cross(fe_3x3_a_mer, fe_5x5_a_mer) + fe_3x3_a

            fe_5x5_a = self.cross(fe_5x5_a_mer, fe_3x3_a_mer) + fe_5x5_a

            # Merging features
            fe_merge_a = torch.where(abs(fe_3x3_a) - abs(fe_5x5_a) >= 0, fe_3x3_a, fe_5x5_a)


            return fe_merge_a


class FusionModel(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=96, scale_unetfeats=48,
                 kernel_size=3, reduction=4, bias=False, padding=1, stride=1,
                 dilation=1, groups=1, p_act=None, norm=False):
        super(FusionModel, self).__init__()
        if p_act is None:
            p_act = nn.PReLU()

        # Initial feature extractor
        self.feature_extractor = nn.Conv2d(in_c, n_feat, kernel_size=kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation, groups=groups)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Stage 1–3 Encoders/Decoders
        # self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'CAB')
        # self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'CAB')
        # self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'PAB')
        # self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'PAB')
        # self.stage3_encoder = Encoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'SAB')
        # self.stage3_decoder = Decoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'SAB')

        # # Self-attention modules
        # self.sam1o = SAM(n_feat, kernel_size=3, bias=bias)
        # self.sam2o = SAM(n_feat, kernel_size=3, bias=bias)
        # self.sam3o = SAM(n_feat, kernel_size=3, bias=bias)

        # Channel adaptation + SEA Attention
        self.channel_adapter = nn.Conv2d(n_feat, n_feat, kernel_size=1)
        self.seaab = Block(dim=n_feat, key_dim=72, num_heads=9)  # outputs n_feat*3 channels

        # =============================================================
        # == New 1×1 conv layers for channel alignment ================
        # =============================================================
        # 288→36, for first LCA
        self.reduce36 = nn.Conv2d(n_feat, 36, kernel_size=1, bias=False)
        self.block1 = NormDownsample(36, 36, use_norm=norm)  # 输入图像的第二个下采样模块
        self.block2 = NormDownsample(36, 72, use_norm=norm)  # 输入图像的第三个下采样模块
        self.block3 = NormDownsample(72, 144, use_norm=norm)  # 输入图像的第三个下采样模块

        self.block4 = NormUpsample(144, 72, use_norm=norm)  # 输入图像的第三个下采样模块
        self.block5 = NormUpsample(72, 36, use_norm=norm)  # 输入图像的第三个下采样模块
        self.block6 = NormUpsample(36, 36, use_norm=norm)  # 输入图像的第三个下采样模块
        # 36→72, for second LCA
        self.expand36to72 = nn.Conv2d(36, 72, kernel_size=1, bias=False)
        # 72→144, for third LCA
        self.expand72to144 = nn.Conv2d(72, 144, kernel_size=1, bias=False)
        # 144→72, for fourth LCA
        self.compress144to72 = nn.Conv2d(144, 72, kernel_size=1, bias=False)
        # 72→36, for fifth and sixth LCA
        self.compress72to36 = nn.Conv2d(72, 36, kernel_size=1, bias=False)

        # Lightweight Cross Attention modules
        self.ALCA11 = LCA(36, 2)
        self.ALCA12 = LCA(72, 4)
        self.ALCA13 = LCA(144, 8)
        self.ALCA14 = LCA(144, 8)
        self.ALCA15 = LCA(72, 4)
        self.ALCA16 = LCA(36, 2)
        self.ALCA21 = LCA(36, 2)
        self.ALCA22 = LCA(72, 4)
        self.ALCA23 = LCA(144, 8)
        self.ALCA24 = LCA(144, 8)
        self.ALCA25 = LCA(72, 4)
        self.ALCA26 = LCA(36, 2)

        # Final fusion
        # self.final_ch_adapter = nn.Sequential(
        #     nn.Conv2d(36, 256, kernel_size=1),  # 将36通道提升到256
        #     nn.BatchNorm2d(256),
        #     nn.ReLU()
        # )
        self.fusion = nn.Sequential(
            nn.Conv2d(36, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 36, kernel_size=3, padding=1),
            nn.BatchNorm2d(36),
            nn.ReLU()
        )
        self.channel_align = nn.Sequential(
            nn.Conv2d(72, 36, kernel_size=3, padding=1),  # 288→256通道对齐
            nn.BatchNorm2d(36),
            nn.ReLU()
        )

        # 修改最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(36, 3, kernel_size=1),  # 正确通道转换
            nn.Tanh()
        )
    def forward(self, img1, img2):
        # Initial features
        feat1 = F.leaky_relu(self.feature_extractor(img1), negative_slope=0.2)
        feat2 = F.leaky_relu(self.feature_extractor(img2), negative_slope=0.2)

        # U-Net paths + SAM
        # def process_stage(feat, img, enc, dec, sam):
        #     outs = enc(feat)
        #     outs = dec(outs)
        #     return sam(outs[0], img)[0]

        # feat1_1 = process_stage(feat1, img1, self.stage1_encoder, self.stage1_decoder, self.sam1o)
        # feat1_2 = process_stage(feat1, img1, self.stage2_encoder, self.stage2_decoder, self.sam2o)
        # feat1_3 = process_stage(feat1, img1, self.stage3_encoder, self.stage3_decoder, self.sam3o)
        # concat_1 = torch.cat([feat1_1, feat1_2, feat1_3], dim=1)
        feat1 = self.seaab(feat1)

        # feat2_1 = process_stage(feat2, img2, self.stage1_encoder, self.stage1_decoder, self.sam1o)
        # feat2_2 = process_stage(feat2, img2, self.stage2_encoder, self.stage2_decoder, self.sam2o)
        # feat2_3 = process_stage(feat2, img2, self.stage3_encoder, self.stage3_decoder, self.sam3o)
        # concat_2 = torch.cat([feat2_1, feat2_2, feat2_3], dim=1)
        feat2 = self.seaab(feat2)

        # ===========================================
        # Channel alignment + LCA cascade
        # ===========================================
        # Level 1 (36)
        feat1_36 = self.reduce36(feat1)
        feat2_36  = self.reduce36(feat2)
        enc1_1 =  self.block1(feat1_36)
        enc2_1 =  self.block1(feat2_36)
        jump1_1 = feat1_36 
        jump2_1 = feat2_36 
        
        

        feat1_2 = self.ALCA11(enc1_1, enc2_1)
        feat2_2 = self.ALCA21(enc2_1, enc1_1)
        jump1_2 = feat1_2
        jump2_2 = feat2_2
        feat1_2 = self.block2(feat1_2)
        feat2_2 = self.block2(feat2_2)

        feat1_3 = self.ALCA12(feat1_2, feat2_2)
        feat2_3 = self.ALCA22(feat2_2, feat1_2)
        jump1_3 = feat1_3
        jump2_3 = feat2_3
        feat1_3 = self.block3(feat1_2)
        feat2_3 = self.block3(feat2_2)

        feat1_4 = self.ALCA13(feat1_3, feat2_3)
        feat2_4 = self.ALCA23(feat2_3, feat1_3)

        feat1_5 = self.ALCA14(feat1_4, feat2_4)
        feat2_5 = self.ALCA24(feat2_4, feat1_4)

        feat2_3 = self.block4(feat2_4, jump2_3)
        f_dec_3 = self.block4(feat1_5, jump1_3)
        f_dec_2 = self.ALCA15(f_dec_3, feat2_3)
        ff_2 = self.ALCA25(feat2_3, f_dec_3)

        ff_2 = self.block5(ff_2, jump2_2)
        f_dec_2  = self.block5(f_dec_3, jump1_2)

        f_dec_1 = self.ALCA16(f_dec_2, ff_2)
        ff_1 = self.ALCA26(ff_2, f_dec_2)

        f_dec_1 = self.block6(f_dec_1, jump1_1)
        ff_1 = self.block6(ff_1, jump2_1)

       

        # feat1_final = self.final_ch_adapter(feat1_36_dec)
        # feat2_final = self.final_ch_adapter(feat2_36_dec)

        # 调整特征图尺寸
        # feat1_36_dec = F.interpolate(feat1_final, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        # feat2_36_dec = F.interpolate(feat2_final, size=feat2.shape[2:], mode='bilinear', align_corners=False)

        # 最终融合
        fused_output1 = self.fusion(f_dec_1)
        fused_output2 = self.fusion(ff_1)
        fused = torch.cat([fused_output1, fused_output2], dim=1)
        fused = self.channel_align(fused)  # 通道对齐到256
        fused = self.final_conv(fused)     # 256→3通道转换
        return fused