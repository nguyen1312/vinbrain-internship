import sys
sys.path.append('../../')
from lib import *

if __name__ == "__main__":
    resnet = torchvision.models.resnet.resnet50(pretrained = True)
    layer1 = nn.Sequential(*list(resnet.children()))[:4]
    down_blocks = []
    for bottleneck in list(resnet.children()):
        if isinstance(bottleneck, nn.Sequential):
            down_blocks.append(bottleneck)
    # print(len(down_blocks))
    print(down_blocks[2])
    a = torch.rand(4, 3, 512, 512)
    a1 = layer1(a)
    print(a1.size())