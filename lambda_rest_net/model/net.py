import torch
import torch.nn as nn
from lambda_rest_net.model.lambda_res_net import lambda_resnet50

class Net(nn.Module):
    def __init__(self, in_channels=3):
        super(Net, self).__init__()
        # Paramter
        res_net_in = 32
        res_net_out = 2048
        age_nurones = 64

        # Image
        self.conv_in = nn.Conv2d(in_channels=in_channels, out_channels=res_net_in, kernel_size=1, stride=1, bias=False)
        self.resnet = lambda_resnet50(num_classes=res_net_out, channel_in=res_net_in)

        # Age
        self.age = nn.Linear(in_features=1, out_features=age_nurones)
        self.activate = nn.ReLU()

        # Regressor
        self.reg_1 = self.build_regessor(in_features=age_nurones+res_net_out, out_features=1500)
        self.reg_2 = self.build_regessor(in_features=1000, out_features=1000)
        self.reg_3 = self.build_regessor(in_features=1000, out_features=1000)


        # Out 
        self.out = nn.Linear(in_features=1000, out_features=1)



    def build_regessor(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.LeakyReLU(),
            nn.Dropout(),
        )


    def forward(self, x, y):
        x = x.float()
        x = x / 255.0
        y = y.float()
        x = self.conv_in(x)
        x = self.resnet(x)

        y = self.age(y.view(-1, 1))

        x = torch.cat((x, y), 1)
        x = self.activate(x)

        x = self.reg_1(x)
        x = self.reg_2(x)
        x = self.reg_3(x)

        x = self.out(x)
        return x
