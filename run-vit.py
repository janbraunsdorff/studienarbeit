import vit.model.config as conf
import torchvision.transforms as transforms
import torchvision
import torch
from vit.model.vit import ViT
from vit.trainer import Trainer
from vit.mussure import benchmark
import sys
import os
from vit.preprocessing.processing import process_store_image_train, process_store_image_val, load_data


if os.getenv('USER') == "janbraunsdorff":
    path_to_data = '/Users/janbraunsdorff/Studienarbeit-projekt/data'
elif os.getenv('USER') == "janbrauns": 
    path_to_data = '/home/janbrauns/data'


path_to_validation_annotation = path_to_data + "/validation.csv"
path_to_training_annotation = path_to_data + "/training.csv"
path_to_validatoin_data = path_to_data + '/boneage-validation-dataset/'
path_to_training_data = path_to_data + '/boneage-training-dataset/'
path = path_to_data + '/pickel/v3'

transform = transforms.Compose(
    [
        transforms.Resize((conf.image_size, conf.image_size)),
        transforms.ToTensor(),
    ]
)

#trainset = torchvision.datasets.CIFAR100(root='./vit/data', train=True, download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=conf.batch_szie, shuffle=True)

#testset = torchvision.datasets.CIFAR100(root='./vit/data', train=False, download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=conf.batch_szie, shuffle=False)

def pre():
    print("proccess images train", end=" ")
    benchmark(process_store_image_train, path_to_training_annotation, path_to_training_data, path)
    sys.stdout.flush()
    print("proccess images val", end=" ")
    benchmark(process_store_image_val, path_to_validation_annotation, path_to_validatoin_data, path)
    sys.stdout.flush()

#pre()

print('init model...', end=' ')
model = ViT()
model = model.to(conf.device) 
print('**done** \nload data...', end='')

train_loader, val_loader =  benchmark(load_data, path, conf.batch_szie)

print('**done** \ncreate Trainer...', end='')
trainer = Trainer(model, train_loader, val_loader)

print('**done** \nfit...')
trainer.fit()
print(trainer.history)