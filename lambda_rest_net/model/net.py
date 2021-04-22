import torch
import torch.nn as nn
from lambda_rest_net.model.lambda_res_net import LambdaResNet

class Net(nn.Module):
    def __init__(self, in_channels=3):
        super(Net, self).__init__()
        # Paramter
        res_net_in = 32
        res_net_out = 2000
        age_nurones = 96

        # Image
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=res_net_in, kernel_size=1, stride=1, bias=False)
        self.resnet = LambdaResNet(in_channels=32, layers=[3, 4, 6, 3], num_classes=2000)

        # Age
        self.age = nn.Linear(in_features=1, out_features=age_nurones)
        self.activate = nn.ReLU()

        # Regressor
        self.reg_1 = self.build_regessor(in_features=age_nurones+res_net_out, out_features=1500)
        self.reg_2 = self.build_regessor(in_features=1500, out_features=1000)


        # Out 
        self.out = nn.Linear(in_features=1000, out_features=1)

        



    def build_regessor(self, in_features, out_features):
        nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(in_features=in_features, out_features=out_features),
        )


    def forward(self, x, y):
        x = x / 255.0
        x = self.conv_in(x)
        x = self.resnet(x)

        y = self.age(y.view(-1, 1))
        y = self.activate(y)

        x = torch.cat((x, y), 1)

        x = self.reg_1(x)
        x = self.reg_2(x)

        x = self.out(x)
        return x
