import torch
from torch import nn, einsum
from einops import rearrange
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

from lambda_rest_net.model.lambda_layer import LambdaLayer

class LambdaBottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None) :
        super(LambdaBottleneck, self).__init__()
        # Parameter
        self.stride = stride
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 =  nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = LambdaLayer(dim=width, dim_out=width, r = 23, dim_k=4, heads=2, dim_u=2)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # functions
        self.downsample = downsample



    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out