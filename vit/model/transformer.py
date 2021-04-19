import torch.nn as nn
import torch
import vit.model.config as conf
from vit.model.multilayerPerceptron import MultilayerPerceptron


class Transformer(nn.Module):
    def __init__(self, patch_size):
        super(Transformer, self).__init__()
        self.norm_1 = nn.BatchNorm1d(num_features=256, eps=1e-6)
        self.norm_2 = nn.BatchNorm1d(num_features=256, eps=1e-1)
        self.mha = nn.MultiheadAttention(embed_dim=conf.project_dim, num_heads=conf.num_heads, dropout=0.1).to(conf.device)
        self.mlp = MultilayerPerceptron(dropout_rate=0.1, layers=conf.hidden_layers)
        self.to(conf.device)



    def forward(self, encoded_patches):
        print(encoded_patches.shape)
        x1 = self.norm_1(encoded_patches)
        attention_output = self.mha(x1,x1,x1)
        print('attention: ', attention_output[0].shape)
        x2 = attention_output[0] + encoded_patches
        x3 = self.norm_2(x2)
        x3 = self.mlp(x3)
        ret = x3 + x2

        return ret, attention_output[1]