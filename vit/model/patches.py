import vit.model.config as conf
import torch.nn as nn
import torch

class Patches(nn.Module):
  def __init__(self, patch_size):
    super(Patches, self).__init__()
    self.patch_size = patch_size
    self.to(conf.device)


  def forward(self, images):
    # 256, 3, 72, 72
    batch = images.size()[0]
    patches = images.unfold(1, 3, 3).unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
    patches = patches.squeeze(1)
    patches = patches.resize(batch, 12*12 ,3 * self.patch_size* self.patch_size)
    return patches


class PatchEncoder (nn.Module):
  def __init__(self, num_patches, project_dim):
    super(PatchEncoder, self).__init__()
    self.num_patches = num_patches
    self.projection = nn.Linear(in_features=108, out_features=project_dim)
    self.prostional_embedding = nn.Embedding(num_embeddings=144, embedding_dim=project_dim)
    self.to(self.device)



  def forward(self, patch):
    # patch = B x 144 x 108
    positions =  torch.arange(start=0, end=self.num_patches, step=1).to(self.device)
    # positions = 144
    p1 = self.projection(patch)
    # p1 = B x 144 x 64
    p2 = self.prostional_embedding(positions)
    # p2 = 144, 64
    encoded =  p1+p2
    # encoded = B x 144 x 64
    return encoded