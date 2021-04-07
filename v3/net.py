import torch
import torch.nn as nn
import numpy as np

import sys
from v3.mussure import benchmark
import torchvision.transforms as transforms


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class MnistModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.inception_v3 = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
        self.inception_v3.fc = Identity()
   
        self.dense32 = nn.Linear(1, 64)

        self.dense1000_1 = nn.Linear(2048+64, 1000)
        self.dense1000_2 = nn.Linear(1000, 1000)
        self.dense1000_4 = nn.Linear(1000, 1)

        self.drop_1 = nn.Dropout()
        self.drop_2 = nn.Dropout()
        self.drop_3 = nn.Dropout()


        self.relu = torch.nn.ReLU()
        self.device = device

        self.aug =  aug = transforms.Compose(
            [
                transforms.RandomRotation(degrees=20),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=5),
                ], p=0.4),
                transforms.RandomErasing(p=0.5)
            ]
        )

    def agument(self, x):
        if self.training:
            x = self.aug(x)
        return x


    def forward(self, x, y):
        x = x.float()
        x = x / 255.0
        x = self.agument(x)

        y = y.float()
        x = self.inception_v3(x)
        if self.training:
            x = x.logits
        y = self.dense32(y.unsqueeze(1).float())


        x = torch.cat((x, y), 1)


        x = self.relu(x)
        x = self.dense1000_1(x)
        x = self.relu(x)
        x = self.drop_1(x)

        x = self.dense1000_2(x)
        x = self.relu(x)
        x = self.drop_2(x)

        x = self.dense1000_4(x)
        x = self.relu(x)

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
        images = images.to(self.device)                     # move to device
        labels = labels.to(self.device)                     # move to device
        sex = sex.to(self.device)

        out = self(images, sex)                             # generate Predictions
        out = out.reshape(list(out.size())[0])         # reshape from [batch, 1] -> [batch]

        loss = nn.MSELoss()(out, labels.float())       # calculate loss
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
        print("epoch [{:3}]".format(epoch), end=', ')
        print("val_loss: \u001B[31m{:8.4f}\x1b[0m".format(result['val_loss']), end=', ')
        print("val_acc: \u001B[31m{:8.4f}\x1b[0m".format(result['val_acc']), end=', ')
        print("c4 : \u001B[31m{:4}/{} ({:8.4f}%)\x1b[0m".format(result['c4'],  1425, (result['c4'] /1425)  * 100), end=", ")
        print("c12: \u001B[31m{:4}/{} ({:8.4f}%)\x1b[0m".format(result['c12'], 1425, (result['c12'] /1425) * 100), end=", ")
        print("c24: \u001B[31m{:4}/{} ({:8.4f}%)\x1b[0m".format(result['c24'], 1425, (result['c24'] /1425) * 100), end=", ")


def fit(epochs, lr, betas,  model, train_loader, val_loader, stop_after, opt_func=torch.optim.Adam):
    optimizer = opt_func(model.parameters(), lr=lr,  betas=betas)
    sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, threshold=0.001, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-10, verbose=False)
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
            torch.save(model, 'model/v3_trash.pth')
            eps_without_no_new_optim = 0
        else:
            eps_without_no_new_optim = eps_without_no_new_optim + 1

        print("max_c12: \u001B[35m{:4}\x1b[0m, eps_without_no_new_optim: \u001B[35m{:2}\x1b[0m".format(min_loss, eps_without_no_new_optim))
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
    return torch.sum(torch.abs(torch.sub(outputs, labels))) / len(labels)

def correct(out, lbl, trashold):
    out = torch.round(out)
    lbl = torch.round(lbl)
    ab = torch.abs(lbl - out).detach().cpu()
    return len(np.where(ab.numpy() <= trashold)[0])

