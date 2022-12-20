import sys
sys.path.append('../../')

from lib import *
from Conv import ConvBlock

#  Upsample -> ConvBlock -> ConvBlock
class UpBlock(nn.Module):    
    def __init__(self, chan_in, 
                       chan_out, 
                       upconv_in = None, 
                       upconv_out = None,
                ):
        super().__init__()
        if upconv_in == None:
            upconv_in = chan_in
        if upconv_out == None:
            upconv_out = chan_out
        self.upsample = nn.ConvTranspose2d(upconv_in, upconv_out, kernel_size = 2, stride = 2)
        self.conv = nn.Sequential(ConvBlock(chan_in, chan_out),
                                  ConvBlock(chan_out, chan_out))

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x1 = torch.cat([x1, x2], 1)
        x1 = self.conv(x1)
        return x1
# Test
if __name__ == "__main__":
    a = torch.rand(4, 2048, 16, 16)
    b = torch.rand(4, 1024, 32, 32)
    convNet = UpBlock(2048, 1024)
    out = convNet(a, b)
    print(out.size())
    print(out)

