import vit.model.config as conf
import torch
import torch.nn as nn
import numpy as np

class Trainer():
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=conf.lr)
        self.history = []
        self.train_data = train_data
        self.val_data = val_data

    def fit(self):
        for i in range(conf.num_epoch):
            epoch_loss_train, epoch_acc_train = self.train()
            epoch_loss_val, epoch_acc_val = self.eval()
            self.history.append((epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

    def train(self):
        self.model.train()
        eps = []
        for batch in self.train_data:
            x , y = batch
            x = x.to(conf.device)
            y = y.to(conf.device)

            out = self.model(x)
            loss = nn.CrossEntropyLoss()(out, y)

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            x = x.cpu()
            y = y.cpu()
            out = out.cpu()

            acc = self.accuracy(out, y)
            eps.append({'val_loss': loss.item(), 'val_acc': acc.item()})

        batch_losses = [x['val_loss'] for x in eps]
        epoch_loss = np.average(batch_losses)
        batch_accs = [x['val_acc'] for x in eps]
        epoch_acc = (np.sum(batch_accs) / 50000.0) * 100.0
        print("train-loss: {:.4f}, train-acc: {:.4f}%".format(epoch_loss, epoch_acc), end=' | ')

        return epoch_loss, epoch_acc

    def eval(self):
        self.model.eval()
        eps = []
        for batch in self.val_data:
            x, y = batch

            x = x.to(conf.device)
            y = y.to(conf.device)

            out = self.model(x)
            loss = nn.CrossEntropyLoss()(out, y)

            out.cpu()
            x = x.cpu()
            y = y.cpu()

            acc = self.accuracy(out, y)
            eps.append({'val_loss': loss.item(), 'val_acc': acc.item()})

        batch_losses = [x['val_loss'] for x in eps]
        epoch_loss = np.average(batch_losses)
        batch_accs = [x['val_acc'] for x in eps]
        epoch_acc = (np.sum(batch_accs) / 10000.0) * 100.0
        print("test-loss: {:.4f}, test-acc: {:.4f}%".format(epoch_loss, epoch_acc))

        return epoch_loss, epoch_acc


    def accuracy(self, out, labels):
        return len(out) - torch.count_nonzero(torch.argmax(out, dim=1)- torch.abs(labels))