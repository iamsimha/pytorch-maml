import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

torch.manual_seed(1000)
torch.cuda.manual_seed(10000)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


"""
Implementing MAML requires model to take in the parameters, in addition to inputs.
Hence we use pytorch's functional API.
This code is inspired from https://github.com/tristandeleu/pytorch-meta
"""


class MetaConv2d(nn.Conv2d):
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
            weight = params.get("weight", None)
            bias = params.get("bias", None)
        else:
            weight = params.get(self.module_prefix + ".weight", None)
            bias = params.get(self.module_prefix + ".bias", None)
        return F.conv2d(
            input,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class MetaLinear(nn.Linear):
    def forward(self, input, params):
        if params is None:
            params = OrderedDict(self.named_parameters())
            weight = params.get("weight", None)
            bias = params.get("bias", None)
        else:
            weight = params.get(self.module_prefix + ".weight", None)
            bias = params.get(self.module_prefix + ".bias", None)

        return F.linear(input, weight, bias)


class MetaBatchNorm2d(nn.BatchNorm2d):
    """
    This snippet is copied from
    https://github.com/pytorch/pytorch/blob/a89d1ed5496dccce310b3dd08a417b5de71e8332/torch/nn/modules/batchnorm.py#L76
    """

    def forward(self, input, params=None):
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if params is None:
            params = OrderedDict(self.named_parameters())
            weight = params.get("weight", None)
            bias = params.get("bias", None)
        else:
            weight = params.get(self.module_prefix + ".weight", None)
            bias = params.get(self.module_prefix + ".bias", None)
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            weight,
            bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )
