import torch
import torch.nn as nn
from torchvision.models.inception import InceptionA, InceptionB, InceptionC
import vit.model.config as conf

class Regreesor(nn.Module):
    def __init__(self):
        super(Regreesor, self).__init__()
        self.activate = nn.LeakyReLU()
        self.norm = nn.BatchNorm2d(num_features=1)
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3))
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3))
        self.max_Pool = nn.MaxPool2d(kernel_size=(2,2))

        self.inceptionA_1 = InceptionA(128, 256)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv_1(x)
        x = self.activate(x)
        x = self.conv_2(x)
        x = self.activate(x)
        x = self.conv_2(x)
        x = self.activate(x)
        x = self.max_Pool(x)

        x = self.inceptionA_1(x)
        print(x.shape)

        raise Exception('Regressor end')
        



