import torch
import torch.nn as nn
from torchvision.models.inception import InceptionA, InceptionB, InceptionC, InceptionD, InceptionE
import vit.model.config as conf

class Regreesor(nn.Module):
    def __init__(self):
        super(Regreesor, self).__init__()
        self.activate = nn.LeakyReLU()
        self.norm = nn.BatchNorm2d(num_features=1)
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=2)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=2)
        self.max_Pool = nn.MaxPool2d(kernel_size=(2,2))

        self.inceptionA_1 = InceptionA(64, 32)
        self.inceptionA_2 = InceptionA(256, 32)
        self.inceptionA_3 = InceptionA(256, 64)

        self.inceptionB_1 = InceptionB(288)

        self.inceptionA_4 = InceptionC(768, 128)
        self.inceptionA_5 = InceptionC(768, 160)
        self.inceptionA_6 = InceptionC(768, 160)
        self.inceptionA_7 = InceptionC(768, 192)

        self.inceptionD_8 = InceptionD(768)
        self.inceptionE_9 = InceptionE(1280)
        self.inceptionE_10 = InceptionE(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.faltten = nn.Flatten()

        self.dense = nn.Linear(2048, 1000)
        self.drop = nn.Dropout()
        self.out = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv_1(x)
        x = self.activate(x)
        x = self.conv_2(x)
        x = self.activate(x)
        x = self.conv_3(x)
        x = self.activate(x)
        x = self.max_Pool(x)
        
        x = self.inceptionA_1(x)
        x = self.inceptionA_2(x)
        x = self.inceptionA_3(x)

        x = self.inceptionB_1(x)

        x = self.inceptionA_4(x)
        x = self.inceptionA_5(x)
        x = self.inceptionA_6(x)
        x = self.inceptionA_7(x)

        x = self.inceptionD_8(x)
        x = self.inceptionE_9(x)
        x = self.inceptionE_10(x)

        x = self.avgpool(x)
        x = self.drop(x)

        x = self.faltten(x)
        x = self.dense(x)
        x = self.activate(x)
        x = self.drop(x)
        x = self.out(x)


        return x



