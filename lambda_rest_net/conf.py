import torch

batch_size = 32
lr = 1e-4
num_epoch = 100

res_file = 'lambda-resnet.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)