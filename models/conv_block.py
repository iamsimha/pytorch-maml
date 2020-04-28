import torch.nn as nn
from models.metamodules import MetaBatchNorm2d, MetaConv2d

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                MetaConv2d(in_channels, out_channels, kernel_size=kernel_size),
                MetaBatchNorm2d(out_channels, track_running_stats=False),
                nn.ReLU(),
            ]
        )

    def forward(self, inputs, params=None):
        out = inputs
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                out = layer(out)
            else:
                out = layer(out, params)
        return out
