import torch.nn as nn
import torch
import sys

class Encoder(nn.Module):
    def __init__(self, num_featuers=200, kernel_size=(3,3), stride=1):
        super(Encoder, self).__init__()

        self.conv_1_1 = nn.Conv2d(in_channels=3,  out_channels=64, kernel_size=kernel_size, padding=1)
        self.conv_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=1)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv_2_1 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=kernel_size, padding=1)
        self.conv_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=1)

        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv_3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=1)
        self.conv_3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernel_size, padding=1)

        self.max_pool_3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv_4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kernel_size, padding=1)
        self.conv_4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel_size, padding=1)

        self.max_pool_4 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.shrink_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1))
        self.shrink_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,1))
        self.shrink_3 = nn.Conv2d(in_channels=128, out_channels=64,  kernel_size=(1,1))
        self.shrink_4 = nn.Conv2d(in_channels=64,  out_channels=16,  kernel_size=(1,1))
        self.shrink_5 = nn.Conv2d(in_channels=16,  out_channels=3,  kernel_size=(1,1))

        self.flatten = nn.Flatten()
        self.featuers = nn.Linear(in_features=16*16*16, out_features=1000)

        self.activation = nn.LeakyReLU()

    
    def forward(self, x):
        x = self.conv_1_1(x)
        x = self.activation(x)
        x = self.conv_1_2(x)
        x = self.activation(x)

        x = self.max_pool_1(x)

        x = self.conv_2_1(x)
        x = self.activation(x)
        x = self.conv_2_2(x)
        x = self.activation(x)

        x = self.max_pool_2(x)

        x = self.conv_3_1(x)
        x = self.activation(x)
        x = self.conv_3_2(x)
        x = self.activation(x)

        
        x = self.max_pool_3(x)

        x = self.conv_4_1(x)
        x = self.activation(x)
        x = self.conv_4_2(x)
        x = self.activation(x)

        x = self.max_pool_4(x)


        x = self.shrink_1(x)
        x = self.activation(x)
        x = self.shrink_2(x)
        x = self.activation(x)
        x = self.shrink_3(x)
        x = self.activation(x)
        x = self.shrink_4(x)
        x = self.activation(x)
        x = self.shrink_5(x)
        x = self.activation(x)

        x = self.flatten(x)
        x = self.featuers(x)

        return x