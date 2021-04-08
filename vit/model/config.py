import torch

lr = 1e-5
weight_decay = 1e-5
batch_szie = 32
num_epoch = 10_000
image_size = 256
patch_size = 16
num_patches = (image_size // patch_size) ** 2
project_dim = 128
num_heads = 8
hidden_layers = [
    (project_dim, project_dim*2), 
    (project_dim*2, project_dim*2), 
    (project_dim*2, project_dim*2), 
    (project_dim*2, project_dim)
]
transformer_layers = 16
# mlp_head_units = [4096, 2048, 2048, 1024]
res_file = 'result-first.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)