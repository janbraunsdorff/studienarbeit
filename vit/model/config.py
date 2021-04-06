import torch

lr = 1e-3
weight_decay = 1e-5
batch_szie = 128
num_epoch = 1000
image_size = 256
patch_size = 16
num_patches = (image_size // patch_size) ** 2
project_dim = 128
num_heads = 4
hidden_layers = [
    (project_dim, project_dim*2), 
    (project_dim*2, project_dim*2), 
    (project_dim*2, project_dim*2), 
    (project_dim*2, project_dim)
]
transformer_layers = 16
mlp_head_units = [4096, 2048, 1024]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)