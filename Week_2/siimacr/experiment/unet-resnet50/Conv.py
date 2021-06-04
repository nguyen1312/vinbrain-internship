import sys
sys.path.append('../../')

from lib import *
# Conv -> BN -> ReLU
class ConvBlock(nn.Module):
    def __init__(self, chan_in, 
                       chan_out, 
                       padding = 1, 
                       kernel_size = 3, 
                       stride = 1
                ):
        super().__init__()
        self.conv = nn.Conv2d(chan_in, 
                              chan_out, 
                              padding = padding, 
                              kernel_size = kernel_size, 
                              stride = stride)
        self.bn = nn.BatchNorm2d(chan_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Test
if __name__ == "__main__":
    a = torch.rand(4, 3, 512, 512)
    convNet = ConvBlock(3, 1)
    out = convNet(a)
    print(out.size())
    print(out)

