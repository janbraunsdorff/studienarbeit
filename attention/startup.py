from attention.model import Regressor, fit, evaluate
from attention.processing import process_store_image_train, process_store_image_val, load_data
from attention.mussure import benchmark

from torchsummary import summary

import torch
import time
import os
import sys

if os.getenv('USER') == "janbraunsdorff":
    path_to_data = '/Users/janbraunsdorff/Studienarbeit-projekt/data'
elif os.getenv('USER') == "janbrauns": 
    path_to_data = '/home/janbrauns/data'


path_to_validation_annotation = path_to_data + "/validation.csv"
path_to_training_annotation = path_to_data + "/training.csv"
path_to_validatoin_data = path_to_data + '/boneage-validation-dataset/'
path_to_training_data = path_to_data + '/boneage-training-dataset/'
path = path_to_data + '/pickel/v3'


def pre_process():
    print("proccess images train", end=" ")
    benchmark(process_store_image_train, path_to_training_annotation, path_to_training_data, path)
    sys.stdout.flush()
    print("proccess images val", end=" ")
    benchmark(process_store_image_val, path_to_validation_annotation, path_to_validatoin_data, path)
    sys.stdout.flush()

def run(batch_size, lr):
    print("load data: ", end= "")
    train_loader, val_loader =  benchmark(load_data, path, batch_size)
    print("**done**")
    print("get device: ", end="")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("init model: ", end="")
    sys.stdout.flush()
    model = Regressor(device=device)
    model.to(device)
    
    sys.stdout.flush()
    print("first guess: ", end="")
    model.eval()
    result = evaluate(model, val_loader)
    print("epoch [{}]".format('init'), end=' ,')
    print("val_loss: \u001B[31m{:.4f}\x1b[0m".format(result['val_loss']), end=', ')
    print("val_acc: \u001B[31m{:.4f}\x1b[0m".format(result['val_acc']), end=', ')
    print("c4 : \u001B[31m{}/{} ({:.4f}%)\x1b[0m".format(result['c4'],  1425, (result['c4'] /1425)  * 100), end=", ")
    print("c12: \u001B[31m{}/{} ({:.4f}%)\x1b[0m".format(result['c12'], 1425, (result['c12'] /1425) * 100), end=", ")
    print("c24: \u001B[31m{}/{} ({:.4f}%)\x1b[0m".format(result['c24'], 1425, (result['c24'] /1425) * 100))
    sys.stdout.flush()
    
    sys.stdout.flush()
    h1 = fit(100_000, lr, model, train_loader, val_loader, 200)
    print("res: ", h1)

