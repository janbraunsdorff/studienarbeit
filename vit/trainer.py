import vit.model.config as conf
import torch
import torch.nn as nn
import numpy as np
import time
import sys

class Trainer():
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=conf.lr)
        self.sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='max', factor=0.8, patience=10, threshold=0.001, 
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10, verbose=True
        )
        self.history = []
        self.train_data = train_data
        self.val_data = val_data

    def fit(self):
        best_score = 0
        for i in range(conf.num_epoch):
            epoch_loss_train, epoch_acc_train = self.train(i)
            epoch_loss_val, epoch_acc_val, score = self.eval()
            self.history.append((epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val))

            if score > best_score:
                print(' | save model...', end=' ')
                torch.save(self.model, 'model/vit.pth')
                best_score = score
                print('**done**', end='')
            sys.stdout.flush()

            print(' | save model...', end=' ')
            torch.save(self.model, 'model/vit-bad.pth')
            best_score = score
            print('**done**')
            sys.stdout.flush()


    def train(self, epoch):
        self.model.train()
        eps = []
        sum = 0
        start = time.time()

        for index, batch in enumerate(self.train_data):
            x , y , z = batch
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
            loss = loss.cpu()

            acc = self.accuracy(out, y)
            c4 = self.correct(out, y, 4)
            c12 = self.correct(out, y, 12)
            c24 = self.correct(out, y, 24)

            eps.append({'val_loss': loss.item(), 'val_acc': acc.item(), 'c4' : c4, 'c12' : c12,'c24' : c24})


            batch_losses = [x['val_loss'] for x in eps]
            epoch_loss = np.average(batch_losses)
            batch_accs = [x['val_acc'] for x in eps]
            epoch_acc = np.average(batch_accs)

            c4 = [x['c4'] for x in eps]
            c4 = np.sum(c4) / 126.11
            c12 = [x['c12'] for x in eps]
            c12 = np.sum(c12) / 126.11
            c24 = [x['c24'] for x in eps]
            c24 = np.sum(c24) / 126.11


            print('\rEpoche: {:3} [Learn] ({:3}/{:3}) loss: {:8.4f}, acc: {:6.4f}, c4: {:6.2f}% c12: {:6.2f}% c24: {:6.2f}% score: {:.4f}'.format(epoch+1, index + 1, len(self.train_data),epoch_loss, epoch_acc, c4, c12, c24,(c4*2+c12+c24*0.5)), end='')
            sys.stdout.flush()
        end = time.time()

        batch_losses = [x['val_loss'] for x in eps]
        epoch_loss = np.average(batch_losses)
        batch_accs = [x['val_acc'] for x in eps]
        epoch_acc = np.average(batch_accs) 

        c4 = [x['c4'] for x in eps]
        c4 = np.sum(c4) / 126.11
        c12 = [x['c12'] for x in eps]
        c12 = np.sum(c12) / 126.11
        c24 = [x['c24'] for x in eps]
        c24 = np.sum(c24) / 126.11
        
        print("\rEpoche: {:3} [Done] {} loss: {:9.4f}, acc: {:9.4f}, c4: {:6.2f}% c12: {:6.2f}% c24: {:6.2f}% score: {:.4f}".format(epoch+1, time.strftime('%M:%S', time.gmtime(end - start)), epoch_loss, epoch_acc, c4, c12, c24, (c4*2+c12+c24*0.5)), end=' | ')
        f = open(conf.res_file, "a")
        f.write(";".join([str(epoch+1), str(time.strftime('%M:%S', time.gmtime(end - start))), str(epoch_loss), str(epoch_acc), str(c4), str(c12), str(c24), str(c4*2+c12+c24*0.5)]) + ';')
        f.close()
        sys.stdout.flush()

        return epoch_loss, epoch_acc

    def eval(self):
        self.model.eval()
        eps = []
        for batch in self.val_data:
            x, y, z = batch

            x = x.to(conf.device)
            y = y.to(conf.device)
            z = z.to(conf.device)

            out = self.model(x, z)
            out = out.reshape(list(out.size())[0])
            loss = nn.MSELoss()(out, y.float()) 

            out = out.cpu()
            x = x.cpu()
            y = y.cpu()
            z = z.cpu()
            loss.cpu()

            acc = self.accuracy(out, y)
            c4 = self.correct(out, y, 4)
            c12 = self.correct(out, y, 12)
            c24 = self.correct(out, y, 24)
            eps.append({'val_loss': loss.item(), 'val_acc': acc.item(), 'c4' : c4, 'c12' : c12,'c24' : c24})


        batch_losses = [x['val_loss'] for x in eps]
        epoch_loss = np.average(batch_losses)
        batch_accs = [x['val_acc'] for x in eps]
        epoch_acc = np.average(batch_accs)

        c4 = [x['c4'] for x in eps]
        c4 = np.sum(c4) / 14.25
        c12 = [x['c12'] for x in eps]
        c12 = np.sum(c12) / 14.25
        c24 = [x['c24'] for x in eps]
        c24 = np.sum(c24) / 14.25

        self.sheduler.step(c4*2+c12+c24*0.5)


        print("[Test] loss: {:9.4f}, acc: {:9.4f}, c4: {:6.2}% c12: {:6.2f}% c24: {:6.2f}% score: {:8.4f}".format(epoch_loss, epoch_acc, c4, c12, c24, (c4*2+c12+c24*0.5)), end='')
        f = open(conf.res_file, "a")
        f.write(";".join([str(epoch_loss), str(epoch_acc), str(c4), str(c12), str(c24), str(c4*2+c12+c24*0.5)]) + '\n')
        f.close()
        sys.stdout.flush()


        return epoch_loss, epoch_acc, c4*2+c12+c24*0.5

    def correct(self, out, lbl, trashold):
        out = torch.round(out)
        lbl = torch.round(lbl)
        ab = torch.abs(lbl - out).detach().cpu()
        return len(np.where(ab.numpy() <= trashold)[0])

    def accuracy(self, out, labels):
        return torch.sum(torch.abs(torch.sub(out, labels))) / len(labels)