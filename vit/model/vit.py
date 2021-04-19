import torch.nn as nn
import vit.model.data_augmentation as augmentation
from vit.model.patches import PatchEncoder, Patches
from vit.model.transformer import Transformer
import vit.model.config as conf
import torch
import numpy as np

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.patches = Patches(patch_size=conf.patch_size)
        self.encode_patches = PatchEncoder(num_patches=conf.num_patches, project_dim=conf.project_dim)
        self.norm_1 = nn.BatchNorm1d(num_features=256, eps=1e-6)
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(p=0.5)

        self.mlp_layer_1 = nn.Linear(in_features=conf.project_dim * conf.num_patches, out_features=4096)
        self.mlp_drop_1 = nn.Dropout(p=0.1)
        self.mlp_layer_2 = nn.Linear(in_features=4096, out_features=2048)
        self.mlp_drop_2 = nn.Dropout(p=0.1)

        self.activate = nn.GELU()
        self.dense32 = nn.Linear(in_features=1, out_features=256)

        self.dense1000_1 = nn.Linear(2048+256, 1)
        #self.dense1000_2 = nn.Linear(1000, 1000)
        #self.dense1000_4 = nn.Linear(1000, 1)

        self.drop_1 = nn.Dropout()
        self.drop_2 = nn.Dropout()
        self.drop_3 = nn.Dropout()

        self.to(conf.device)
        

        self.transformers = []
        for i in range(conf.transformer_layers):
            self.transformers.append(Transformer(conf.patch_size))


    def forward(self, x, sex):
        # x =  B x 3 x 72 x 72
        x = x / 255.0
        sex = sex.float()
        # x =  B x 3 x 72 x 72
        aug = augmentation.data_augmentation(x)
        # aug = B x 3 x 72 x 72
        patches = self.patches(x)
        # patch =  B x 144 x 108
        encoding = self.encode_patches(patches)
        # encoding = B x 144 x 64

        for t in self.transformers:
            encoding, att = t(encoding)
        # encode_patches = B x 144 x 64 

        print(att.squeeze(1).shape)
        print(att.squeeze(1).reshape(64, 16, 16).shape)
        print(torch.argmax(att[0]))
        raise Exception('nö')

        representation = self.norm_1(encoding)
        # representation_1:  torch.Size([256, 144, 64])
        representation = self.flatten(representation)
        # representation_2:  torch.Size([256, 9216])
        representation = self.drop(representation)
        # representation_3:  torch.Size([256, 9216])

        y = self.mlp_layer_1(representation)
        y = self.activate(y)
        y = self.mlp_drop_1(y)

        y = self.mlp_layer_2(y)
        y = self.activate(y)
        y = self.mlp_drop_2(y)


        sex = self.dense32(sex.unsqueeze(1).float())
        x = torch.cat((y, sex), 1)


        x = self.activate(x)
        x = self.drop_1(x)
        x = self.dense1000_1(x)
        x = self.activate(x)

        #x = self.dense1000_2(x)
        #x = self.activate(x)
        #x = self.drop_2(x)

        #x = self.dense1000_4(x)
        #x = self.activate(x)
        
        return x