
class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        
    def forward(self, x):
        return self.bridge(x)