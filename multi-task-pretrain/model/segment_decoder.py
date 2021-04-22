import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, 1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, filters, n_classes):
        super(Decoder, self).__init__()
        self.decoder4 = DecoderBlock(in_channels, filters[3])
        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1])
        self.decoder1 = DecoderBlock(filters[1], filters[0])

        self.final_deconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)


    def forward(self, x):
        # print("x[0].shape: {}".format(x[0].shape))
        # print("x[1].shape: {}".format(x[1].shape))
        # print("x[2].shape: {}".format(x[2].shape))
        # print("x[3].shape: {}".format(x[3].shape))
        # print("x[4].shape: {}".format(x[4].shape))

        d4 = self.decoder4(x[4]) + x[3]
        d3 = self.decoder3(d4) + x[2]
        d2 = self.decoder2(d3) + x[1]
        d1 = self.decoder1(d2)

        out = self.final_deconv1(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        
        # return F.sigmoid(out)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
