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



## Convolution and Attention Fusion Module  (CAFM)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=(1,1,1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=(3,3,3), stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1,1,1), bias=bias)
        self.fc = nn.Conv3d(3*self.num_heads, 9, kernel_size=(1,1,1), bias=True)

        self.dep_conv = nn.Conv3d(9*dim//self.num_heads, dim, kernel_size=(3,3,3), bias=True, groups=dim//self.num_heads, padding=1)


    def forward(self, x):
        b,c,h,w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0,2,3,1) 
        f_all = qkv.reshape(f_conv.shape[0], h*w, 3*self.num_heads, -1).permute(0, 2, 1, 3) 
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        #local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9*x.shape[1]//self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv) # B, C, H, W
        out_conv = out_conv.squeeze(2)


        # global SA
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output =  out + out_conv

        return output



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
                     kernel_size=3, reduction=4, bias=False,  padding=1, stride=1, dilation=1, groups=1, p_act=None,):
        super(FusionModel, self).__init__()
        if p_act is None:
            p_act = nn.PReLU()
        # in_c = args.in_ch
        self.feature_extractor = nn.Conv2d(in_c, 96, kernel_size=kernel_size, stride=stride, padding=padding, 
                      dilation=dilation, groups=groups)  # 3通道输入，64通道输出
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'CAB')
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'CAB')


        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'PAB')
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'PAB')

        self.stage3_encoder = Encoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'SAB')
        self.stage3_decoder = Decoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'SAB')
        
        
        self.sam1o = SAM(n_feat, kernel_size=3, bias=bias)
        self.sam2o = SAM(n_feat, kernel_size=3, bias=bias)
        self.sam3o = SAM(n_feat, kernel_size=3, bias=bias)
        self.channel_adapter = nn.Conv2d(288, 96, kernel_size=1) 
        self.seaab = Block(dim=96 * 3, key_dim= 72, num_heads= 9)
        
        self.channel_compress = nn.Conv2d(96 * 3 * 2, 256, kernel_size=1)  # 576 -> 256
        self.attn = Attention(dim=256, num_heads=8, bias=True)  # 调整输入通道数
        self.multi_ker_size = MultiKerSize(64)  # 使用 64 通道作为输入
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(128, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(48, out_c, kernel_size=3, stride=1, padding=1),  # 最终输出3通道（RGB图像）
            nn.Tanh()
        )

    def forward(self, img1, img2):
        # Multi-scale feature extraction
        feat1 = F.leaky_relu(self.feature_extractor(img1), negative_slope=0.2)
        feat2 = F.leaky_relu(self.feature_extractor(img2), negative_slope=0.2)
      
        
        feat1 = self.maxpool(feat1)
        feat2 = self.maxpool(feat2)

        
        
        feat1_1 = self.stage1_encoder(feat1)
        feat1_1 = self.stage1_decoder(feat1_1)
        # 获取SAM模块返回的元组中的第一个元素（假设需要的是这个张量进行拼接等后续操作，你可根据实际情况调整）
        feat1_1 = self.sam1o(feat1_1[0], img1)[0]
        feat1_2 = self.stage2_encoder(feat1)
        feat1_2 = self.stage2_decoder(feat1_2)
        feat1_2 = self.sam2o(feat1_2[0], img1)[0]
        feat1_3 = self.stage3_encoder(feat1)
        feat1_3 = self.stage3_decoder(feat1_3)
        feat1_3 = self.sam3o(feat1_3[0], img1)[0]
        concat_1 = torch.cat([feat1_1, feat1_2, feat1_3], dim=1)
        feat1 =self.seaab(concat_1)

        feat2_1 = self.stage1_encoder(feat2)
        feat2_1 = self.stage1_decoder(feat2_1)
        feat2_1 = self.sam1o(feat2_1[0], img2)[0]
        feat2_2 = self.stage2_encoder(feat2)
        feat2_2 = self.stage2_decoder(feat2_2)
        feat2_2 = self.sam2o(feat2_2[0], img2)[0]
        feat2_3 = self.stage3_encoder(feat2)
        feat2_3 = self.stage3_decoder(feat2_3)
        feat2_3 = self.sam3o(feat2_3[0], img2)[0]
        concat_2 = torch.cat([feat2_1, feat2_2, feat2_3], dim=1)
        feat2 =self.seaab(concat_2)

        # feat1 = self.MFEM(feat1)
        # feat2 = self.MFEM(feat2)
        # feat3 = self.MFEM(feat3)

        # feat1 = self.attn(feat1)
        # feat2 = self.attn(feat2)
        # feat3 = self.attn(feat3)
        # # Apply MultiKerSize
        # feat1 = self.multi_ker_size(feat1)
        # feat2 = self.multi_ker_size(feat2)
        # feat3 = self.multi_ker_size(feat3)

        # Concatenate features
        fused_features = torch.cat([feat1, feat2], dim=1)
        # fused_features =self.psa(fused_features)
        # Fusion
        fused_features = self.channel_compress(fused_features)  # 新增压缩
        fused_features = self.attn(fused_features)
        fused_output = self.fusion(fused_features)

        # Generate final output
        output = self.output_layer(fused_output)

        return output

