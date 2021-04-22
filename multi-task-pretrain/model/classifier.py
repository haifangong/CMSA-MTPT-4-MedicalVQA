import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

class Classifier(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Classifier, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, n_classes),
            # nn.Softmax(dim=1)
            )
        self._init_weight()

    def forward(self, x):
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
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


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, args):
        super(SimpleClassifier, self).__init__()
        activation_dict = {'relu': nn.ReLU()}
        try:
            activation_func = activation_dict[args.activation]
        except:
            raise AssertionError(args.activation + " is not supported yet!")
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            activation_func,
            nn.Dropout(args.dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits