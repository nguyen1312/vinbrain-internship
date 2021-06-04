import sys
sys.path.append('../../')
from lib import *
from Conv import ConvBlock

class Bridge(nn.Module):
    def __init__(self, chan_in, chan_out):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(chan_in, chan_out),
            ConvBlock(chan_out, chan_out)
        )

    def forward(self, x):
        return self.bridge(x)

# Test
if __name__ == "__main__":
    a = torch.rand(4, 3, 512, 512)
    convNet = Bridge(3, 1)
    out = convNet(a)
    print(out.size())
    print(out)