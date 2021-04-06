import torch.nn as nn
import numpy as np
import torch
from autoencoder.mussure import benchmark
import sys

class Autodecoder(nn.Module):
    def __init__(self, encoder, decoder, device, lr=1e-3):
        super(Autodecoder, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder
        self.device = device
        
        # loss
        self.loss = nn.BCELoss()

        # optim
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # train
        self.history = []


    def forward(self, x):
        x = x.float() 
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch):
        img, lbl, sex = batch
        img = img.to(self.device)
        img = img.float()

        out = self(img)
        loss = self.loss(out, img)

        img.cpu()
        return loss

    def epoch(self, data_loader):
        self.optimizer.step()
        self.optimizer.zero_grad()
        epoch_loss = 0

        for batch in data_loader:
            loss = self.training_step(batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        for batch in data_loader:
            epoch_loss += np.abs(self.training_step(batch).item())

        epoch_loss = epoch_loss / 12611
        self.history.append(epoch_loss)
        return epoch_loss

    def fit(self, epochs, train_data):
        for i in range(epochs):
            loss = benchmark(self.epoch, train_data)
            print('[Epoch]: {} [Loss]: {:.4f}'.format(i, loss))
            sys.stdout.flush()


        return self.history