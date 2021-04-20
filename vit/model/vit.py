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
        print('5', patches.shape)
        raise Exception('nö')

        # patch =  B x 144 x 108
        encoding = self.encode_patches(patches)
        # encoding = B x 144 x 64

        for t in self.transformers:
            encoding = t(encoding)
        # encode_patches = B x 144 x 64 

        t = torch.mean(encoding, 2)

        max_value = torch.max(t, dim=1)[0]
        t = t / max_value.view(16,1)
        print(t.shape, '\n', t)


        return x

    def scale_maks(self, mask):
        t = mask.repeat_interleave(2).view(-1, 16, 32).repeat(1, 1, 2).view(-1,32,32)
        t = t.repeat_interleave(2).view(-1, 32,64).repeat(1,1,2).view(-1,64,64)
        t = t.repeat_interleave(2).view(-1, 64,128).repeat(1,1,2).view(-1,128,128)
        t = t.repeat_interleave(2).view(-1, 128,256).repeat(1,1,2).reshape(-1,256,256)
        return t