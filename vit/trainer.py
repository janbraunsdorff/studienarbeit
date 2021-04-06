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
            epoch_loss_train, epoch_acc_train = self.train(i)
            epoch_loss_val, epoch_acc_val = self.eval()
            self.history.append((epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

    def train(self, epoch):
        self.model.train()
        eps = []
        sum = 0
        for index, batch in enumerate(self.train_data):
            x , y , z = batch
            sum += x.shape[0]
            x = x.to(conf.device)
            y = y.to(conf.device)
            z = z.to(conf.device)

            out = self.model(x, z)
            out = out.reshape(list(out.size())[0])
            loss = nn.MSELoss()(out, y.float()) 

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            x = x.cpu()
            y = y.cpu()
            z = z.cpu()
            out = out.cpu()

            acc = self.accuracy(out, y)
            c4 = self.correct(out, y, 4)
            c12 = self.correct(out, y, 12)
            c24 = self.correct(out, y, 24)
            eps.append({'val_loss': loss.item(), 'val_acc': acc.item(), 'c4' : c4, 'c12' : c12,'c24' : c24})


            batch_losses = [x['val_loss'] for x in eps]
            epoch_loss = np.average(batch_losses)
            batch_accs = [x['val_acc'] for x in eps]
            epoch_acc = (np.average(batch_accs) / sum) * 100.0

            c4 = [x['c4'] for x in eps]
            c4 = np.sum(c4)
            c12 = [x['c12'] for x in eps]
            c12 = np.sum(c12)
            c24 = [x['c24'] for x in eps]
            c24 = np.sum(c24)

            print('\rEpoche: {} [Training] ({}/{}) loss: {:.4f}, acc: {:.4f}%, c1: {:.2f}% c12: {:.2f}% c24: {:.2f}%'.format(epoch, index + 1, len(self.train_data),epoch_loss, epoch_acc, c4/126.11, c12/126.11, c24/126.11), end='')

        batch_losses = [x['val_loss'] for x in eps]
        epoch_loss = np.average(batch_losses)
        batch_accs = [x['val_acc'] for x in eps]
        epoch_acc = (np.sum(batch_accs) / 12611) * 100.0

        c4 = [x['c4'] for x in eps]
        c4 = np.sum(c4)
        c12 = [x['c12'] for x in eps]
        c12 = np.sum(c12)
        c24 = [x['c24'] for x in eps]
        c24 = np.sum(c24)
        print("[Train] loss: {:.4f}, acc: {:.4f}, c1: {} c12: {} c24: {}".format(epoch_loss, epoch_acc, c4, c12, c24), end=' | ')

        return epoch_loss, epoch_acc

    def eval(self):
        self.model.eval()
        eps = []
        sum = 0
        for batch in self.val_data:
            x, y, z = batch

            x = x.to(conf.device)
            y = y.to(conf.device)
            z = z.to(conf.device)

            out = self.model(x, z)
            loss = nn.MSELoss()(out, y.float()) 

            out = out.cpu()
            x = x.cpu()
            y = y.cpu()
            z = z.cpu()

            acc = self.accuracy(out, y)
            c4 = self.correct(out, y, 4)
            c12 = self.correct(out, y, 12)
            c24 = self.correct(out, y, 24)
            eps.append({'val_loss': loss.item(), 'val_acc': acc.item(), 'c4' : c4, 'c12' : c12,'c24' : c24})


        batch_losses = [x['val_loss'] for x in eps]
        epoch_loss = np.average(batch_losses)
        batch_accs = [x['val_acc'] for x in eps]
        epoch_acc = (np.sum(batch_accs) / 12611) * 100.0

        c4 = [x['c4'] for x in eps]
        c4 = np.sum(c4)
        c12 = [x['c12'] for x in eps]
        c12 = np.sum(c12)
        c24 = [x['c24'] for x in eps]
        c24 = np.sum(c24)
        print("[Test] loss: {:.4f}, acc: {:.4f}, c1: {} c12: {} c24: {}".format(epoch_loss, epoch_acc, c4, c12, c24))

        return epoch_loss, epoch_acc

    def correct(self, out, lbl, trashold):
        out = torch.round(out)
        lbl = torch.round(lbl)
        ab = torch.abs(lbl - out).detach().cpu()
        return len(np.where(ab.numpy() <= trashold)[0])

    def accuracy(self, out, labels):
        return torch.sum(torch.abs(torch.sub(out, labels))) / len(labels)