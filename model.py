import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# Multi-axis Partial Queried Learning Block (MPQLB)
class MPQLB(nn.Module):
    def __init__(self, dim, x=8, y=8, bias=False):
        super(MPQLB, self).__init__()

        partial_dim = int(dim // 4)

        self.hw = nn.Parameter(torch.ones(1, partial_dim, x, y), requires_grad=True)
        self.conv_hw = nn.Conv2d(partial_dim, partial_dim, kernel_size=to_2tuple(3), padding=1, groups=partial_dim, bias=bias)

        self.ch = nn.Parameter(torch.ones(1, 1, partial_dim, x), requires_grad=True)
        self.conv_ch = nn.Conv1d(partial_dim, partial_dim, kernel_size=3, padding=1, groups=partial_dim, bias=bias)

        self.cw = nn.Parameter(torch.ones(1, 1, partial_dim, y), requires_grad=True)
        self.conv_cw = nn.Conv1d(partial_dim, partial_dim, kernel_size=3, padding=1, groups=partial_dim, bias=bias)

        self.conv_4 = nn.Conv2d(partial_dim, partial_dim, kernel_size=to_2tuple(1), bias=bias)

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(3), padding=1, groups=dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias),
        )

    def forward(self, x):
        input_ = x
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        # hw
        x1 = x1 * self.conv_hw(F.interpolate(self.hw, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # ch
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2 * self.conv_ch(
            F.interpolate(self.ch, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # cw
        x3 = x3.permute(0, 2, 1, 3)
        x3 = x3 * self.conv_cw(
            F.interpolate(self.cw, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)

        x4 = self.conv_4(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.norm2(x)
        x = self.mlp(x) + input_

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super(BasicLayer, self).__init__()
        self.blocks = nn.ModuleList([MPQLB(dim, dim) for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=to_2tuple(3), padding=1, bias=bias),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=to_2tuple(3), padding=1, bias=bias),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# Supervised Cross-scale Transposed Attention Module (SCTAM)
class SCTAM(nn.Module):
    def __init__(self, dim, up_scale=2, bias=False):
        super(SCTAM, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

        self.up = nn.PixelShuffle(up_scale)

        self.qk_pre = nn.Conv2d(int(dim // (up_scale ** 2)), 3, kernel_size=to_2tuple(1), bias=bias)
        self.qk_post = nn.Sequential(LayerNorm2d(3),
                                     nn.Conv2d(3, int(dim * 2), kernel_size=to_2tuple(1), bias=bias))

        self.v = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias)
        )

        self.conv = nn.Conv2d(dim, dim, kernel_size=to_2tuple(3), padding=1, groups=dim, bias=bias)

        self.norm =LayerNorm2d(dim)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(3), padding=1, groups=dim, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qk = self.qk_pre(self.up(x))
        fake_image = qk
        qk = self.qk_post(qk).reshape(b, 2, c, -1).transpose(0, 1)
        q, k = qk[0], qk[1]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        v = self.v(x)
        v_ = v.reshape(b, c, h*w)

        attn = (q @ k.transpose(-1, -2)) * self.alpha
        attn = attn.softmax(dim=-1)
        x = (attn @ v_).reshape(b, c, h, w) + self.conv(v)

        x = self.norm(x)
        x = self.proj(x)

        return x, fake_image


class PIFM(nn.Module):
    def __init__(self, channel, reduction=8, bias=False):
        super(PIFM, self).__init__()

        hidden_features = int(channel // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.a = nn.Sequential(
            nn.Conv2d(channel, hidden_features, kernel_size=to_2tuple(1), bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_features, channel * 2, kernel_size=to_2tuple(1), bias=bias),
            nn.Softmax(dim=1)
        )
        self.t = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=to_2tuple(3), padding=1, groups=channel, bias=bias),
            nn.Conv2d(channel, hidden_features, kernel_size=to_2tuple(1), bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_features, channel * 2, kernel_size=to_2tuple(1), bias=bias),
            nn.Sigmoid()
        )

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        in_feats_ = in_feats.view(B, 2, C, H, W)
        x = torch.sum(in_feats_, dim=1)

        a = self.a(self.avg_pool(x))
        t = self.t(x)
        j = torch.mul((1 - t), a) + torch.mul(t, in_feats)

        j = j.view(B, 2, C, H, W)
        j = torch.sum(j, dim=1)
        return j


class EMPFNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=4, dim=24, depths=(4, 4, 4, 2, 2)):
        super(EMPFNet, self).__init__()

        self.patch_embed = nn.Conv2d(in_channel, dim, kernel_size=to_2tuple(3), padding=1)
        self.skip1 = BasicLayer(dim, depths[0])

        self.down1 = Downsample(dim)
        self.skip2 = BasicLayer(int(dim * 2 ** 1), depths[1])

        self.down2 = Downsample(int(dim * 2 ** 1))
        self.latent = BasicLayer(int(dim * 2 ** 2), depths[2])

        self.up1 = Upsample(int(dim * 2 ** 2))
        self.sctam1 = SCTAM(int(dim * 2 ** 1), up_scale=2)
        self.pifm1 = PIFM(int(dim * 2 ** 1))
        self.layer4 = BasicLayer(int(dim * 2 ** 1), depths[3])

        self.up2 = Upsample(int(dim * 2 ** 1))
        self.sctam2 = SCTAM(dim, up_scale=1)
        self.pifm2 = PIFM(dim)
        self.layer5 = BasicLayer(dim, depths[4])

        self.patch_unembed = nn.Conv2d(dim, out_channel, kernel_size=to_2tuple(3), padding=1, bias=False)

    def forward_features(self, x):

        x = self.patch_embed(x)
        skip1 = x

        x = self.down1(x)
        skip2 = x

        x = self.down2(x)
        x = self.latent(x)
        x = self.up1(x)
        x, fake_image_x4 = self.sctam1(x)

        x = self.pifm1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.up2(x)
        x, fake_image_x2 = self.sctam2(x)

        x = self.pifm2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)

        return x, fake_image_x4, fake_image_x2

    def forward(self, x, only_last=False):
        input_ = x
        _, _, h, w = input_.shape

        x, fake_image_x4, fake_image_x2 = self.forward_features(x)
        K, B = torch.split(x, [1, 3], dim=1)

        x = K * input_ - B + input_
        x = x[:, :, :h, :w]

        if only_last:
            return x
        else:
            return x, fake_image_x4, fake_image_x2


if __name__ == '__main__':
    x = torch.randn((1, 3, 512, 512)).cuda()
    net = EMPFNet().cuda()

    from thop import profile, clever_format
    flops, params = profile(net, (x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
