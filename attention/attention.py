import torch.nn as nn
import torch 
import sys

class Self_Attn(nn.Module):
    def __init__(self, in_dim, reduction):
        super().__init__() 
        self.out_channel = in_dim//reduction
        
        # Construct the conv layers
        # 512 -> 64
        self.f = nn.Conv2d(in_channels = in_dim , out_channels = self.out_channel, kernel_size= 1) # query
        self.g = nn.Conv2d(in_channels = in_dim , out_channels = self.out_channel , kernel_size= 1) # key
        self.h = nn.Conv2d(in_channels = in_dim , out_channels = in_dim            , kernel_size= 1) # value
        
        self.row_softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channesl, h, w = x.size()

        query = self.f(x)
        query = query.reshape(-1, self.out_channel, h*w)


        key = self.g(x)
        key = key.reshape(-1, self.out_channel, h*w)

        value = self.h(x)
        value = value.reshape(-1, channesl, h*w)
        

        query = torch.transpose(query, 1, 2)
        attention_map = torch.matmul(query, key)
        attention_map = self.row_softmax(attention_map)

        attention_feature_map = torch.matmul(value, attention_map)
        attention_feature_map = attention_feature_map.reshape(-1, channesl, h, w)

        o = self.gamma * attention_feature_map
        o = o + x
        return o