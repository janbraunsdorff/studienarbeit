import torch

lr = 1e-3
weight_decay = 1e-4
batch_szie = 256
num_epoch = 100
image_size = 72
patch_size = 6
num_patches = (image_size // patch_size) ** 2
project_dim = 64
num_heads = 4
hidden_layers = [
    (project_dim, project_dim*2), 
    (project_dim*2, project_dim)
]
transformer_layers = 8
num_classes = 1000
mlp_head_units = [2048, 1024]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)