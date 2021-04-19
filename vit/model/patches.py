import vit.model.config as conf
import torch.nn as nn
import torch

class Patches(nn.Module):
  def __init__(self, patch_size):
    super(Patches, self).__init__()
    self.patch_size = patch_size
    self.to(conf.device)


  def forward(self, images):
    print('1', images.shape)
    batch = images.size()[0]
    patches = images.unfold(1, 1, 1).unfold(2, conf.patch_size, conf.patch_size).unfold(3, conf.patch_size, conf.patch_size)
    print('2', patches.shape)
    patches = patches.squeeze(1)
    print('3', patches.shape)
    patches = patches.resize(batch, conf.num_patches, conf.patch_size* conf.patch_size)
    print('4'. patches.shape)
    raise Exception('nö')
    return patches


class PatchEncoder (nn.Module):
  def __init__(self, num_patches, project_dim):
    super(PatchEncoder, self).__init__()
    self.num_patches = num_patches
    self.projection = nn.Linear(in_features=conf.patch_size * conf.patch_size, out_features=project_dim)
    self.prostional_embedding = nn.Embedding(num_embeddings=conf.patch_size * conf.patch_size, embedding_dim=project_dim)
    self.to(conf.device)



  def forward(self, patch):
    # patch = B x 256 x 256
    positions =  torch.arange(start=0, end=self.num_patches, step=1).to(conf.device)
    # positions = 144
    p1 = self.projection(patch)
    # p1 = B x 144 x 64
    p2 = self.prostional_embedding(positions)
    # p2 = 144, 64
    encoded =  p1+p2
    # encoded = B x 144 x 64
    return encoded