import torch.nn as nn
import torch 
from attention.attention import Self_Attn


class Featuer_Extraction(nn.Module):
    def __init__(self):
        super().__init__()

        # torch.Size([1, 3, 400, 400]) -> torch.Size([1, 128, 129, 129])

        self.activation = nn.LeakyReLU()

        self.l1_con1 = nn.Conv2d(in_channels=1,  out_channels=128, kernel_size=(3,3), stride=1)
        self.l1_con3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1)
        self.l1_pool = nn.MaxPool2d(kernel_size=(3,3))
        self.l1_norm = nn.BatchNorm2d(128)

        self.l3_con1 = nn.Conv2d(in_channels=128,  out_channels=256, kernel_size=(3,3), stride=1)
        self.l3_con3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1)
        self.l3_pool = nn.MaxPool2d(kernel_size=(3,3))
        self.l3_norm = nn.BatchNorm2d(256)

        self.l2_con1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3))
        self.l2_con2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3))
        self.l2_pool = nn.MaxPool2d(kernel_size=(3,3))
        self.l2_norm = nn.BatchNorm2d(512)

    def forward(self,x):
        x = self.l1_con1(x)
        x = self.activation(x)
        x = self.l1_con3(x)
        x = self.activation(x)
        x = self.l1_pool(x)
        x = self.l1_norm(x)

        x = self.l3_con1(x)
        x = self.activation(x)
        x = self.l3_con3(x)
        x = self.activation(x)
        x = self.l3_pool(x)
        x = self.l3_norm(x)


        x = self.l2_con1(x)
        x = self.activation(x)
        x = self.l2_con2(x)
        x = self.activation(x)
        x = self.l2_pool(x)
        x = self.l2_norm(x)


        return x