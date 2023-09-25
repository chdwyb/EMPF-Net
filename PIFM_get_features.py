import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple


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


        import matplotlib.pyplot as plt
        from torchvision.transforms import ToPILImage
        from torchvision.utils import save_image

        # transmission map
        for i in range(t.shape[1]):
            sc = plt.imshow(1 / t[0][i].cpu())
            # sc.set_cmap('jet')
            plt.axis('off')
            plt.savefig('I://RS_dehazing_fig/1_t/'+str(i+1)+'.png', bbox_inches='tight', pad_inches=0.0)
            print(f'{i+1}: {a[0][i]}')

        j = torch.mul((1 - t), a) + torch.mul(t, in_feats)

        j = j.view(B, 2, C, H, W)
        j = torch.sum(j, dim=1)

        for i in range(j.shape[1]):
            sc = plt.imshow(j[0][i].cpu())
            # sc.set_cmap('jet')
            plt.axis('off')
            plt.savefig('I://RS_dehazing_fig/j/'+str(i+1)+'.png', bbox_inches='tight', pad_inches=0.0)

        return j