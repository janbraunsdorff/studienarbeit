import lambda_rest_net.conf as conf
import torchvision.transforms as transforms
import torchvision
import torch
from lambda_rest_net.model.net import Net
from lambda_rest_net.trainer_res import Trainer
from vit.mussure import benchmark
import sys
import os
from lambda_rest_net.preprocessing.processing import process_store_image_train, process_store_image_val, load_data


if os.getenv('USER') == "janbraunsdorff":
    path_to_data = '/Users/janbraunsdorff/Studienarbeit-projekt/data'
elif os.getenv('USER') == "janbrauns": 
    path_to_data = '/home/janbrauns/data'


path_to_validation_annotation = path_to_data + "/validation.csv"
path_to_training_annotation = path_to_data + "/training.csv"
path_to_validatoin_data = path_to_data + '/boneage-validation-dataset/'
path_to_training_data = path_to_data + '/boneage-training-dataset/'
path = path_to_data + '/pickel/v3'


def pre():
    print("proccess images train", end=" ")
    benchmark(process_store_image_train, path_to_training_annotation, path_to_training_data, path)
    sys.stdout.flush()
    print("proccess images val", end=" ")
    benchmark(process_store_image_val, path_to_validation_annotation, path_to_validatoin_data, path)
    sys.stdout.flush()

pre()

print('init model...', end=' ')
model = Net(in_channels=1)
model = model.to(conf.device) 
print('**done** \nload data...', end='')

train_loader, val_loader =  benchmark(load_data, path, conf.batch_size)

print('**done** \ncreate Trainer...', end='')
trainer = Trainer(model, train_loader, val_loader)

print('**done** \nfit...')
trainer.fit()
print(trainer.history)