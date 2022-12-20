import sys
sys.path.append('../../')
from lib import *
from UpBlock import UpBlock
from Bridge import Bridge

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained = True)
        # conv -> batchnorm -> reLU
        self.module0 = nn.Sequential(*list(resnet.children()))[:3] 
        # max pool layer in resnet
        self.module0pool = list(resnet.children())[3] 
        # collect bottleneck in resnet to use as block module in Unet
        encoder = []
        # there are 4 module nn.Sequential in Resnet50
        for module in list(resnet.children()):
            if isinstance(module, nn.Sequential):
                encoder.append(module)
        self.encoder = nn.ModuleList(encoder)
        self.bridge = Bridge(2048, 2048)

        self.decoder = nn.ModuleList([
            UpBlock(2048, 1024),
            UpBlock(1024, 512),
            UpBlock(512, 256),
            UpBlock(192, 128, 256, 128),
            UpBlock(67, 64, 128, 64)
        ])
        self.outLayer = nn.Conv2d(64, 1, kernel_size = 1, stride = 1)

    def forward(self, x):
        tempStorage = dict()
        tempStorage["t_0"] = x
        # x.size(): 3x512x512
        x = self.module0(x)
        # x.size(): 64x256x256
        tempStorage["t_1"] = x
        x = self.module0pool(x)
        # x.size():64x128x128
        for idx, module in enumerate(self.encoder, start = 2):
            x = module(x)
            # 2, 3, 4, 5
            if idx == 5: 
                continue
            tempStorage[f"t_{idx}"] = x

        x = self.bridge(x)

        for idx, module in enumerate(self.decoder, start = 1):
            match_indice = 5 - idx
            temp_key = f"t_{match_indice}"
            x = module(x, tempStorage[temp_key])
        x = self.outLayer(x)
        return x

# Test
if __name__ == "__main__":
    a = torch.rand(4, 3, 512, 512)
    net = UNet()
    print(net)
    out = net(a)
    print(out.size())
    print(out)
