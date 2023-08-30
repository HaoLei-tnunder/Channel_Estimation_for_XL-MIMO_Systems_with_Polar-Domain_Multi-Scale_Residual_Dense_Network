import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channel):
        super(ASPP,self).__init__()        
        depth = in_channel
        self.mean = nn.AdaptiveAvgPool2d(1)       # 自适应均值池化
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        # self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 4, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        # atrous_block18 = self.atrous_block18(x)
 
        # cat = torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1)
        cat = torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12], dim=1)      
        out = self.conv_1x1_output(cat)
        return out
