import torch
import torch.nn as nn

class Convert(nn.Module):
    def __init__(self, image_size, backbone_output_dim, os, v_dim):
        super(Convert, self).__init__()
        size = int(image_size / os)
        in_dim = size * size * backbone_output_dim
        self.linear = nn.Linear(in_dim, v_dim)
        self._init_weight()

    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.linear(x)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class GAPConvert(object):
    """docstring for GAPConvert"""
    def __init__(self):
        super(GAPConvert, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.avg_pool(x)
        out = torch.flatten(x, 1)
        return out