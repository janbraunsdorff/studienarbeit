import torch

lr = 1e-6
weight_decay = 1e-5
batch_szie = 16
num_epoch = 10_000
image_size = 256
patch_size = 16
num_patches = (image_size // patch_size) ** 2
project_dim =  16*12
num_heads = 16
hidden_layers = [
    (project_dim, project_dim*2), 
    (project_dim*2, project_dim*2), 
    (project_dim*2, project_dim*2), 
    (project_dim*2, project_dim)
]
transformer_layers = 10
res_file = 'result-inc-vit.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

# clear && git add . && git commit -m "test" && git push && nvidia-smi && tail -n 5 result-trans-8-small-hidden.csv
# clear &&  nvidia-smi && tail -n 5 result-trans-8-small-hidden.csv
# clear && git pull && nohup python3 run-vit.py
# clear && git pull && python3 run-vit.py
