import os
import random
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import itertools
from itertools import repeat
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from torch.utils.data import BatchSampler

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def generate_mask(data, p=0.5, remain=0):
    B, T, C = data.shape

    ts = data[0, :, 0]
    et_num = ts[~torch.isnan(ts)].size(0) - remain
    total, num = et_num * C, round(et_num * C * p)

    while True:
        i_mask = np.zeros(total)
        i_mask[random.sample(range(total), num)] = 1
        i_mask = i_mask.reshape(et_num, C)
        if 1 not in i_mask.sum(axis=0) and 0 not in i_mask.sum(axis=0):
            break
        break

    i_mask = torch.tensor(i_mask, dtype=torch.bool)
    mask = i_mask.unsqueeze(0).repeat(B, 1, 1)

    return mask

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, best_score=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def get_accuracy(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    wake_f1 = f1_score(y_true == 0, y_pred == 0)
    n1_f1 = f1_score(y_true == 1, y_pred == 1)
    n2_f1 = f1_score(y_true == 2, y_pred == 2)
    n3_f1 = f1_score(y_true == 3, y_pred == 3)
    rem_f1 = f1_score(y_true == 4, y_pred == 4)

    return acc, f1, kappa, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1

def save_results(folder_path, setting, best, test=0):
    # result save
    acc = best['accuracy']
    f1 = best['f1_score']
    wake_f1 = best['wake_f1']
    n1_f1 = best['n1_f1']
    n2_f1 = best['n2_f1']
    n3_f1 = best['n3_f1']
    rem_f1 = best['rem_f1']
    kappa = best['kappa']
    cm = best['cm']
    print('acc    f1     kappa  wk_f1  n1_f1  n2_f1  n3_f1  rem_f1')
    print('{:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}'
          .format(acc, f1, kappa, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1))
    print(cm)
    print()

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name = 'result-20.txt'
    f = open(os.path.join(folder_path, file_name), 'a')
    if(test == 0):
        f.write('**VALID** ')
    else:
        f.write('TEST ')
    f.write(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time())))
    f.write(setting + "  \n")
    f.write('acc         f1          kappa       wk_f1       n1_f1       n2_f1       n3_f1       rem_f1')
    f.write('\n')
    f.write(
        '{:.8}  {:.8}  {:.8}  {:.8}  {:.8}  {:.8}  {:.8}  {:.8}'
        .format(acc, f1, kappa,
                wake_f1, n1_f1, n2_f1, n3_f1, rem_f1))
    f.write('\n')
    cm_str = np.array2string(cm, separator=' ', threshold=20)
    f.write(cm_str)
    f.write('\n')
    f.close()

def loss_weight_balance(label):
    label, count = np.unique(label, return_counts=True)
    ratio_reciprocal = np.reciprocal(count/count.sum())
    loss_weight = ratio_reciprocal*(len(label)/(ratio_reciprocal.sum()))
    loss_weight = torch.tensor(loss_weight, dtype=torch.float32)
    return loss_weight
