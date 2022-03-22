

from torch import nn
import torch.nn.functional as F
import torch


class ResBlock2d(nn.Module):
    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
    
        self.norm1 = nn.Conv2d(
            in_channels=in_features, out_channels=in_features, kernel_size=1)
        self.norm2 = nn.Conv2d(
            in_channels=in_features, out_channels=in_features, kernel_size=1)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out += x
        return out


class RGBADecoderNet(nn.Module):
    def __init__(self,  c=64, out_planes=4,  num_bottleneck_blocks=1):
        super(RGBADecoderNet, self).__init__()
        self.conv_rgba = nn.Sequential(nn.Conv2d(c, out_planes, kernel_size=3, stride=1,
                                                 padding=1, dilation=1, bias=True))
        self.bottleneck = torch.nn.Sequential()
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module(
                'r' + str(i), ResBlock2d(c, kernel_size=(3, 3), padding=(1, 1)))

    def forward(self, features_weighted_mask_atfeaturesscale_list=[]):
        return torch.sigmoid(self.conv_rgba(self.bottleneck(features_weighted_mask_atfeaturesscale_list.pop(0))))
