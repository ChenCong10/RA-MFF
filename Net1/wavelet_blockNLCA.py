import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# 卷积替换的 Cross-Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads=None, bias=False):
        super(CAB, self).__init__()
        # 拼接两路特征后用两层卷积融合
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        z = torch.cat([x, y], dim=1)  # [B, 2*C, H, W]
        out = self.fuse(z)            # [B, C, H, W]
        return out


# 强化模块 IEL
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=3, padding=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features,
                                kernel_size=3, padding=1,
                                groups=hidden_features, bias=bias)
        self.activation = nn.SiLU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = self.activation(x)
        x = self.project_out(x)
        return x


# 用卷积替换的轻量 Cross Attention (原 LCA)
class LCA(nn.Module):
    def __init__(self, dim, num_heads=None, bias=False):
        super(LCA, self).__init__()
        # 拼接两路后两层卷积
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.gdfn = IEL(dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, y):
        # 拼接融合
        z = torch.cat([x, y], dim=1)      # [B, 2C, H, W]
        fusion = self.conv_fuse(z)        # [B, C, H, W]
        # 残差融合
        out = x + self.alpha * fusion
        # 加上 IEL 模块
        out = out + (1 - self.alpha) * self.gdfn(out)
        return out


# 测试
if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32)
    y = torch.randn(1, 64, 32, 32)
    cab = CAB(dim=64)
    lc = LCA(dim=64)
    out1 = cab(x, y)
    out2 = lc(x, y)
    print("CAB out shape:", out1.shape)
    print("LCA out shape:", out2.shape)
