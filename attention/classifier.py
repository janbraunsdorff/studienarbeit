import torch
import torch.nn as nn

import sys

class Classifier(nn.Module):
    def __init__(self, input):
        super(Classifier, self).__init__()

        self.clas_1 = nn.Linear(input, 1000)
        self.clas_2 = nn.Linear(1000, 1000)
        self.clas_3 = nn.Linear(1000, 1)

        self.batch_1 = nn.BatchNorm1d(1000)
        self.batch_2 = nn.BatchNorm1d(1000)

        self.drop_1 = nn.Dropout()
        self.drop_2 = nn.Dropout()

        self.activation = nn.LeakyReLU()


    def forward(self, x):
        x = self.clas_1(x)
        x = self.drop_1(x)
        x = self.activation(x)
        x = self.batch_1(x)


        x = self.clas_2(x)
        x = self.drop_2(x)
        x = self.activation(x)
        x = self.batch_2(x)


        x = self.clas_3(x)
        return x

