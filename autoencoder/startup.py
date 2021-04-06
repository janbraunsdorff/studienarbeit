import torch
from torchsummary import summary
import time
import os
import sys
from autoencoder.mussure import benchmark
from autoencoder.processing import load_data
from autoencoder.decoder import Decoder
from autoencoder.encoder import Encoder
from autoencoder.autoencoder import Autodecoder
from autoencoder.processing import process_store_image_train, process_store_image_val

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


def run(batch_size, epochs):
    print("load data: ", end= "")
    train_loader, val_loader =  benchmark(load_data, path, batch_size)
    print("**done**")
    print("get device: ", end="")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("init model: ", end="")
    sys.stdout.flush()
    decoder = Decoder()
    encoder = Encoder()
    model = Autodecoder(encoder, decoder, device)
    model.to(device)
    print("**done**")

    print("summarize: ", end="")
    sys.stdout.flush()
    summary(model, input_size=(3, 256, 256))
    sys.stdout.flush()
    print("**done**")
    sys.stdout.flush()


    history = model.fit(epochs, train_loader)
    print("res: ", history)

