import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models.inception import Inception3, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE


class Inception(nn.Module):
    def __init__(self, v3: Inception3):
        super(Inception3, self).__init__()
        # input
        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, weights=v3.Conv2d_2a_3x3.conv.weight)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1, weights=v3.Conv2d_2b_3x3.conv.weight)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1, stride=1, weights=v3.Conv2d_3b_1x1.conv.weight)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, stride=1, weights=v3.Conv2d_4a_3x3.conv.weight)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Block A
        self.Mixed_5b = IncA(192, pool_features=32, model=v3.Mixed_5b)
        self.Mixed_5c = IncA(256, pool_features=64, model=v3.Mixed_5c)
        self.Mixed_5d = IncA(288, pool_features=64, model=v3.Mixed_5d)

        # Block B
        self.Mixed_6a = IncB(288, model=v3.Mixed_6a)

        # Block C
        self.Mixed_6b = IncC(768, channels_7x7=128, model=v3.Mixed_6b)
        self.Mixed_6c = IncC(768, channels_7x7=160, model=v3.Mixed_6c)
        self.Mixed_6d = IncC(768, channels_7x7=160, model=v3.Mixed_6d)
        self.Mixed_6e = IncC(768, channels_7x7=192, model=v3.Mixed_6e)

        # Block D
        self.Mixed_7a = IncD(768, model=v3.Mixed_7a)

        # Block E
        self.Mixed_7b = IncE(1280, model=v3.Mixed_7b)
        self.Mixed_7c = IncE(2048, model=v3.Mixed_7c)

        # out
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()



    def forward(self, x):
        # N x 3 x 299 x 299
        x = self._transform_input(x)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)

        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)

        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)

        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)

        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)

        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)

        return x

    def _transform_input(self, x):
        x = (x * 2) - 1
        return x


class IncA(nn.Module):

    def __init__(self, in_channels: int, pool_features: int, model: InceptionA) -> None:
        super(IncA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1, weights=model.branch1x1.conv.weight)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1, weights=model.branch5x5_1.conv.weight)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2, weights=model.branch5x5_2.conv.weight)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1, weights=model.branch3x3dbl_1.conv.weight)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1, weights=model.branch3x3dbl_2.conv.weight)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1, weights=model.branch3x3dbl_3.conv.weight)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class IncB(nn.Module):

    def __init__(self, in_channels: int, model: InceptionB) -> None:
        super(IncB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2, weight=model.branch3x3.conv.weight)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1, weight=model.branch3x3dbl_1.conv.weight)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1, weight=model.branch3x3dbl_2.conv.weight)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2, weight=model.branch3x3dbl_3.conv.weight)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class IncC(nn.Module):

    def __init__(self, in_channels: int, channels_7x7: int, model: InceptionC) -> None:
        super(IncC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1, weights=model.branch1x1.conv.weight)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1, weights=model.branch7x7_1.conv.weight)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3), weights=model.branch7x7_2.conv.weight)
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0), weights=model.branch7x7_3.conv.weight)

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1, weights=model.branch7x7dbl_1.conv.weight)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0), weights=model.branch7x7dbl_2.conv.weight)
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3), weights=model.branch7x7dbl_3.conv.weight)
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0), weights=model.branch7x7dbl_4.conv.weight)
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3), weights=model.branch7x7dbl_5.conv.weight)

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1, weights=model.branch_pool.conv.weight)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class IncD(nn.Module):

    def __init__(self, in_channels: int, model: InceptionD) -> None:
        super(IncD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1, weights=model.branch3x3_1.conv.weight)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2, weights=model.branch3x3_2.conv.weight)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1, weights=model.branch7x7x3_1.conv.weight)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3), weights=model.branch7x7x3_2.conv.weight)
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0), weights=model.branch7x7x3_3.conv.weight)
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2, weights=model.branch7x7x3_4.conv.weight)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class IncE(nn.Module):

    def __init__( self, in_channels: int, model: InceptionE) -> None:
        super(IncE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1, weights=model.branch1x1.conv.weight)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1, weights=model.branch3x3_1.conv.weight)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1), weights=model.branch3x3_2a.conv.weight)
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0), weights=model.branch3x3_2b.conv.weight)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1, weights=model.branch3x3dbl_1.conv.weight)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1, weights=model.branch3x3dbl_2.conv.weight)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1), weights=model.branch3x3dbl_3a.conv.weight)
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0), weights=model.branch3x3dbl_3b.conv.weight)

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1, weights=model.branch_pool.conv.weight)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding:int = 0, weights: torch.nn.Parameter = None):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=kernel_size, stride=stride, padding=padding)
        if weights:
            self.conv.weight = weights
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
        