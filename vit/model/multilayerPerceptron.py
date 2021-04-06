import torch.nn as nn
import vit.model.config as conf

class MultilayerPerceptron(nn.Module):
  def __init__(self, dropout_rate, layers):
        super(MultilayerPerceptron, self).__init__()
        self.net = []
        for in_feat, out_feat in  layers:
          seq_layer = []
          seq_layer.append(nn.Linear(in_features=in_feat, out_features=out_feat))
          seq_layer.append(nn.Dropout(p=dropout_rate))
          seq_layer.append(nn.GELU())
          self.net.append(nn.Sequential(*seq_layer).to(conf.device))

          self.to(conf.device)


  def forward(self, x):
    for layer in self.net:
      x = layer(x)

    return x