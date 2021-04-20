import torch.nn as nn
import vit.model.data_augmentation as augmentation
from vit.model.patches import PatchEncoder, Patches
from vit.model.transformer import Transformer
from vit.model.regressor import Regreesor
import vit.model.config as conf
import torch
import numpy as np

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.patches = Patches(patch_size=conf.patch_size)
        self.encode_patches = PatchEncoder(num_patches=conf.num_patches, project_dim=conf.project_dim)
        self.norm_1 = nn.BatchNorm1d(num_features=256, eps=1e-6)

        self.trashhold = nn.Parameter(torch.rand(1, requires_grad=True))

        self.regressor = Regreesor()

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
            encoding = t(encoding)
        # encode_patches = B x 144 x 64 

        encoding = self.norm_1(encoding)

        t = torch.mean(encoding, 2)
        max_value = torch.max(t, dim=1)[0]
        t = t / max_value.view(-1,1)
        t = t.view(-1, 16, 16)
        mask = self.scale_maks(t).unsqueeze(1)

        zeros = torch.zeros_like(mask, device=conf.device)
        ones = torch.ones_like(mask, device=conf.device)

        print(mask)
        mask = mask - self.trashhold
        print(mask)
        mask = torch.where(mask > 0, ones, zeros)
        masked_image = mask * x

        x = self.regressor(x, sex)

        return x

    def scale_maks(self, mask):
        t = mask.repeat_interleave(2).view(-1, 16, 32).repeat(1, 1, 2).view(-1,32,32)
        t = t.repeat_interleave(2).view(-1, 32,64).repeat(1,1,2).view(-1,64,64)
        t = t.repeat_interleave(2).view(-1, 64,128).repeat(1,1,2).view(-1,128,128)
        t = t.repeat_interleave(2).view(-1, 128,256).repeat(1,1,2).reshape(-1,256,256)
        return t