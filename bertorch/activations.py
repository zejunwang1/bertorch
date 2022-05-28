# coding=utf-8

import math
import torch
from packaging import version
from torch import nn

def _gelu_python(x):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

if version.parse(torch.__version__) < version.parse("1.4"):
    gelu = _gelu_python
else:
    gelu = nn.functional.gelu


def linear_act(x):
    return x

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {
    "relu": nn.functional.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": torch.tanh,
    "linear": linear_act,
    "sigmoid": torch.sigmoid
}

def get_activation(act_string):
    if act_string in ACT2FN:
        return ACT2FN[act_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
                       act_string, list(ACT2FN.keys())))
