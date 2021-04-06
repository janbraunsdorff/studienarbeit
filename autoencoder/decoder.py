import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, num_featuers=200, kernel_size=(3,3), stride=1):
        super(Decoder, self).__init__()

        self.defeatuer = nn.Linear(in_features=1000, out_features=16*16*16)

        self.expend_1 = nn.ConvTranspose2d(in_channels=16,  out_channels=64,  kernel_size=(1,1))
        self.expend_2 = nn.ConvTranspose2d(in_channels=64,  out_channels=128, kernel_size=(1,1))
        self.expend_3 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=(1,1))
        self.expend_4 = nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=(1,1))

        self.upsample_1 = nn.Upsample(scale_factor=2)

        self.upconv_1_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=kernel_size, padding=1)
        self.upconv_1_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=kernel_size, padding=1)

        self.upsample_2 = nn.Upsample(scale_factor=2)

        self.upconv_2_1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=kernel_size, padding=1)
        self.upconv_2_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=kernel_size, padding=1)

        self.upsample_3 = nn.Upsample(scale_factor=2)

        self.upconv_3_1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=1)
        self.upconv_3_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=kernel_size, padding=1)

        self.upsample_4 = nn.Upsample(scale_factor=2)

        self.upconv_4_1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=1)
        self.upconv_4_2 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=kernel_size, padding=1)


        self.regulize = nn.Sigmoid()
        self.activation = nn.LeakyReLU()


    
    def forward(self, x):
        x = self.activation(x)
        x = self.defeatuer(x)
        x = self.activation(x)

        x = torch.reshape(x, (-1, 16, 8, 8))

        x = self.expend_1(x)
        x = self.activation(x)
        x = self.expend_2(x)
        x = self.activation(x)
        x = self.expend_3(x)
        x = self.activation(x)
        x = self.expend_4(x)
        x = self.activation(x)

        x = self.upsample_1(x)

        x = self.upconv_1_1(x)
        x = self.activation(x)
        x = self.upconv_1_2(x)
        x = self.activation(x)

        x = self.upsample_2(x)


        x = self.upconv_2_1(x)
        x = self.activation(x)
        x = self.upconv_2_2(x)
        x = self.activation(x)

        x = self.upsample_3(x)

        x = self.upconv_3_1(x)
        x = self.activation(x)
        x = self.upconv_3_2(x)
        x = self.activation(x)

        x = self.upsample_4(x)

        x = self.upconv_4_1(x)
        x = self.activation(x)
        x = self.upconv_4_2(x)
        x = self.activation(x)

        x = self.regulize(x)
        return x