import torch
from torch import nn, einsum
from einops import rearrange
from torch import Tensor
from lambda_rest_net.model.lambda_bottleneck import LambdaBottleneck

class LambdaResNet(nn.Module):
    def __init__(self, in_channels=1, base_output=64, layers=[], groups=1, width_per_group= 64, replace_stride_with_dilation=None, zero_init_residual = False, num_classes=2000):
        super(LambdaResNet, self).__init__()
        # Paramter 
        self.inplanes = base_output
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        # Layer
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=base_output, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_1 = nn.BatchNorm2d(num_features=base_output)

        self.layer1 = self._make_layer(LambdaBottleneck, 64, layers[0])
        self.layer2 = self._make_layer(LambdaBottleneck, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(LambdaBottleneck, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        #self.layer4 = self._make_layer(LambdaBottleneck, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.fc = nn.Linear(256 * LambdaBottleneck.expansion, num_classes)


        # Functions
        self.activation = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, LambdaBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, LambdaBottleneck):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block: LambdaBottleneck, planes: int, blocks: int, stride = 1, dilate = False):
        previous_dilation = self.dilation
        downsample = None

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(LambdaBottleneck(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, nn.BatchNorm2d))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(LambdaBottleneck(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)
        

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.activation(x)
        x = self.maxpool(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
       # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x