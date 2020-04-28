import torch.nn as nn
import torch
import torch.nn.functional as F
from models.metamodules import MetaBatchNorm2d, MetaConv2d, MetaLinear
from models.conv_block import ConvBlock
from utils import set_module_prefix


class CNNModel(nn.Module):
    def __init__(self, hparams):
        super(CNNModel, self).__init__()
        self.dim_output = hparams.dim_output
        self.inner_update_lr = hparams.inner_update_lr
        self.meta_lr = hparams.meta_lr
        self.meta_test_num_inner_updates = hparams.meta_test_num_inner_updates
        self.dim_hidden = hparams.dim_hidden
        self.img_size = hparams.img_size
        self.channels = hparams.channels
        self.build_network()
        # This adds the attribute `module_prefix` to each of pytorch modules.
        # This prefix has to be used to get paramters from model.named_parameters()
        set_module_prefix(self)

    def build_network(self):
        self.conv_block = ConvBlock(self.channels, self.dim_hidden, kernel_size=3)
        self.conv_block2 = ConvBlock(self.dim_hidden, self.dim_hidden, kernel_size=3)
        self.conv_block3 = ConvBlock(self.dim_hidden, self.dim_hidden, kernel_size=3)
        self.conv_block4 = ConvBlock(self.dim_hidden, self.dim_hidden, kernel_size=3)
        self.linear = MetaLinear(self.dim_hidden, self.dim_output)

    def forward(self, inp, params=None):
        out = self.conv_block(inp, params)
        out = self.conv_block2(out, params)
        out = self.conv_block3(out, params)
        out = self.conv_block4(out, params)
        out = torch.mean(out, dim=(2, 3))
        return self.linear(out, params)
