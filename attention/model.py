import torch.nn as nn
import torch 

from attention.featureExtraction import Featuer_Extraction
from attention.attention import Self_Attn
from attention.classifier import Classifier

from attention.mussure import benchmark
import sys
import numpy as np

class Regressor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        attention_channels = 512
        gender = 64

        self.fe = Featuer_Extraction()
        self.attention = Self_Attn(attention_channels, 8)
        self.pooling = nn.AvgPool2d(kernel_size=(2,2))
        self.classifier = Classifier(1152 + gender)

        self.deconv1 = nn.Conv2d(in_channels=attention_channels, out_channels=256, kernel_size=(1,1))
        self.deconv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,1))

        self.gender = nn.Linear(1, gender)

        self.activation = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        
    def forward(self,x, y):
        x = x.float()
        x = x / 255.0
        x = self.fe(x)
        x = self.attention(x)
        x = self.pooling(x)
        
        x = self.deconv1(x)
        x = self.activation(x)
        x = self.deconv2(x)
        x = self.activation(x) 
        x = self.flatten(x)

        y = self.gender(y.unsqueeze(1).float())

        x = torch.cat((x, y), 1)
        
        x = self.classifier(x)
        return x

    def training_step(self, batch):
        image, label, sex = batch
        label = label.to(self.device)
        image = image.to(self.device)
        sex = sex.to(self.device)

        out = self(image, sex)
        out = out.reshape(list(out.size())[0])

        loss = nn.MSELoss()(out, label.float())             # calculate loss

        image.cpu()
        label.cpu()
        sex.cpu()

        return loss

    def validation_step(self, batch):
        images, labels, sex = batch                         # split batch in data an label
        images = images.to(self.device)             # move to device
        labels = labels.to(self.device)                     # move to device
        sex = sex.to(self.device)

        out = self(images, sex)                             # generate Predictions
        out = out.reshape(list(out.size())[0])         # reshape from [batch, 1] -> [batch]

        loss = nn.SmoothL1Loss(beta=1)(out, labels.float())       # calculate loss
        acc = accuracy(out, labels)                    # how close we are in avr


        c4 = correct(out, labels, 4)
        c12 = correct(out, labels, 12)
        c24 = correct(out, labels, 24)

        images = images.cpu()
        labels = labels.cpu()
        sex = sex.cpu()
        loss = loss.cpu()
        acc = acc.cpu()

        return {'val_loss': loss.item(), 'val_acc': acc.item(), 'c4' : c4, 'c12' : c12,'c24' : c24}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = np.average(batch_losses)
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = np.average(batch_accs)

        c4 = [x['c4'] for x in outputs]
        c4 = np.sum(c4)

        c12 = [x['c12'] for x in outputs]
        c12 = np.sum(c12)

        c24 = [x['c24'] for x in outputs]
        c24 = np.sum(c24)

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'c4' : c4, 'c12' : c12,'c24' : c24}

    def epoch_end(self, epoch, result):
        print("epoch [{}]".format(epoch), end=' ,')
        print("val_loss: \u001B[31m{:.4f}\x1b[0m".format(result['val_loss']), end=', ')
        print("val_acc: \u001B[31m{:.4f}\x1b[0m".format(result['val_acc']), end=', ')
        print("c4 : \u001B[31m{}/{} ({:.4f}%)\x1b[0m".format(result['c4'],  1425, (result['c4'] /1425)  * 100), end=", ")
        print("c12: \u001B[31m{}/{} ({:.4f}%)\x1b[0m".format(result['c12'], 1425, (result['c12'] /1425) * 100), end=", ")
        print("c24: \u001B[31m{}/{} ({:.4f}%)\x1b[0m".format(result['c24'], 1425, (result['c24'] /1425) * 100), end=", ")

def fit(epochs, lr,  model, train_loader, val_loader, stop_after, opt_func=torch.optim.Adam):
    optimizer = opt_func(model.parameters(), lr=lr)
    sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-15, verbose=True)
    history = []

    min_loss = 0
    eps_without_no_new_optim = 0

    for epoch in range(epochs):
        # Train
        model.train()
        benchmark(run_epoch, model, train_loader, optimizer)
        model.eval()
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        sheduler.step(result['val_loss'])
        history.append(result)

        if (np.max([i['c12'] for i in history]) > min_loss):
            min_loss = result['c12']
            torch.save(model, 'model/attention.pth')
            eps_without_no_new_optim = 0
        else:
            eps_without_no_new_optim = eps_without_no_new_optim + 1

        print("min_loss: \u001B[35m{}\x1b[0m, eps_without_no_new_optim: \u001B[35m{}\x1b[0m".format(min_loss, eps_without_no_new_optim))
        sys.stdout.flush()

        if (eps_without_no_new_optim > stop_after):
            break 

    return history

def run_epoch(model, train_loader, optimizer):
    for batch in train_loader:
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def accuracy(outputs, labels):
    # print(outputs, labels)
    return torch.sum(torch.abs(torch.sub(outputs, labels))) / len(labels)

def correct(out, lbl, trashold):
    out = torch.round(out)
    lbl = torch.round(lbl)
    ab = torch.abs(lbl - out).detach().cpu()
    return len(np.where(ab.numpy() <= trashold)[0])