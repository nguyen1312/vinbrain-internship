import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, chan_in, chan_out, is_downsample = False):
        super(Block, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(chan_in, chan_out, 3, stride = 2, padding = 1, bias = False)
        else:
            self.conv1 = nn.Conv2d(chan_in, chan_out, 3, stride = 1, padding = 1, bias = False)

        self.batchNormLayer1 = nn.BatchNorm2d(chan_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(chan_out, chan_out, 3, stride = 1, padding = 1, bias = False)
        self.batchNormLayer2 = nn.BatchNorm2d(chan_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(chan_in, chan_out, 1, stride = 2, bias = False),
                nn.BatchNorm2d(chan_out)
            )
        elif chan_in != chan_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(chan_in, chan_out, 1, stride = 1, bias=False),
                nn.BatchNorm2d(chan_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.batchNormLayer1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batchNormLayer2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)

class Net(nn.Module):
    def __init__(self, num_classes = 751, reid = False):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.layer1 = nn.Sequential(
            Block(64, 64, is_downsample = False),
            Block(64, 64)
        )

        self.layer2 = nn.Sequential(
            Block(64, 128, is_downsample = True),
            Block(128, 128)
        )
        self.layer3 = nn.Sequential(
            Block(128, 256, is_downsample = True),
            Block(256, 256)
        )
        self.layer4 = nn.Sequential(
            Block(256, 512, is_downsample = True),
            Block(512, 512)
        )
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.reid:
            x = x.div(x.norm(p = 2, dim = 1, keepdim = True))
            return x
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    net = Net()
    x = torch.randn(4, 3, 128, 64)
    y = net(x)
    print(y.size())