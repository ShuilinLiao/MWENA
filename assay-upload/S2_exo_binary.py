# %%
import os
import random
import copy
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.functional import F

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
from collections import Counter
import argparse
import pickle
from net import *
from utils import *
from utils2 import *

np.random.seed(0)

# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--DataPath', type=str, default='../result_upload/', help = 'Path of input data')
parser.add_argument('--OutPath', type=str, default='../result_upload/', help='Output directory')
parser.add_argument('--proj', type=str, default='exoRBase', help='exoRBase, PDAC or ILD')
parser.add_argument('--proj1', type=str, default='BRCA', help='proj1')
parser.add_argument('--proj2', type=str, default='OV', help='proj2')
parser.add_argument('--splRat', type=float, default=0.8, help='split ratio')
parser.add_argument('--repeat', type=int, default=1, help='repeat time')
parser.add_argument('--lmbda', type=float, default=0.001, help='lmbda')
parser.add_argument('--lmbda2', type=float, default=0.01, help='lmbda')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=10, help='batch size for train data')
parser.add_argument('--batch_size_meta', type=int, default=10, help='batch size for meta data')
args = parser.parse_args()

# %%

def seed_torch(seed=0):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()
warnings.filterwarnings("ignore")
torch.manual_seed(123)

def build_model(features):
    model = Net2(features)
    return model

def sigmoid_net(X):
    return torch.sigmoid(torch.matmul(X, W))

class MultiLinearRegression(nn.Module):
    def __init__(self):
        super(MultiLinearRegression, self).__init__()
        self.linear = nn.Linear(featureDim, 1, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)

class MultiLinearRegression1(nn.Module):
    def __init__(self, input, output):
        super(MultiLinearRegression1, self).__init__()
        self.linear = nn.Linear(input, output, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)

class VNet(nn.Module):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

def sigmoid(x):
    x = x.detach().numpy()
    return (1. / (1. + np.exp(-x)))

class Accumulator:  # @save

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def sgd(params, lr, batch_size):  # @save
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def updater(batch_size):
    return sgd([W], lr, batch_size)

def meta_model_update2(meta_model_1, 
                       vnet_1, 
                       x_1, y_1,
                       epoch, lmbda, lmbda2):

    outputs = meta_model_1(x_1).squeeze(-1)
    outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
    cost = nn.BCELoss(reduction='none')
    cost = cost(outputs, y_1)
    cost_v = torch.reshape(cost, (len(cost), 1))
    v_lambda = vnet_1(cost_v.data)
    w1 = abs((v_lambda - v_lambda.max()) / (v_lambda.max() - v_lambda.min()))

    l_f_meta_1 = torch.sum(cost_v * w1) + lmbda * F.l1_loss(meta_model_1.linear.weight,
                                                           target=torch.zeros_like(meta_model_1.linear.weight.detach()),
                                                           size_average=False) + lmbda2 * torch.sum(torch.pow(meta_model_1.linear.weight, 2))

    meta_model_1.zero_grad()
    grads = torch.autograd.grad(l_f_meta_1, (meta_model_1.params()), create_graph=True)
    meta_lr = 1e-3
    meta_model_1.update_params(lr_inner=meta_lr, source_params=grads)
    del grads

def train_auto_meta3(model_1, 
                     vnet_1, 
                     optimizer_model_1, 
                     optimizer_vnet_1, 
                     x_1, y_1, 
                     mx_1, my_1, epoch, lmbda, lmbda2):
    model_1.train()
    my_1 = torch.tensor(my_1)
    y_1 = torch.tensor(y_1)

    ###########################################step 1#################################
    # meta_model_1 = MultiLinearRegression1(featureDim, 1)
    meta_model_1 = build_model(featureDim)
    meta_model_1.load_state_dict(model_1.state_dict())

    meta_model_update2(meta_model_1, vnet_1, x_1, y_1, epoch, lmbda, lmbda2)

    ###########################################step 2#################################
    mx_1 = mx_1.float()
    y_g_hat = meta_model_1(mx_1).squeeze(-1)
    y_g_hat = torch.where(torch.isnan(y_g_hat), torch.zeros_like(y_g_hat), y_g_hat)
    cost = nn.BCELoss(reduction='none')
    l_g_meta1 = cost(y_g_hat, my_1.to(torch.float32))

    l_g_meta1 = l_g_meta1 + lmbda* F.l1_loss(meta_model_1.linear.weight,
                                          target=torch.zeros_like(meta_model_1.linear.weight.detach()),
                                          size_average=True)  + lmbda2 * torch.sum(torch.pow(meta_model_1.linear.weight, 2))

    optimizer_vnet_1.zero_grad()
    l_g_meta1.sum().backward()
    optimizer_vnet_1.step()

    ###########################################step 3#################################
    loss_1, w1 = calculate_weight_loss2(model_1, vnet_1, x_1, y_1)

    loss_1 = loss_1 + lmbda * F.l1_loss(model_1.linear.weight,
                                       target=torch.zeros_like(model_1.linear.weight.detach()),
                                       size_average=False) + lmbda2 * torch.sum(torch.pow(model_1.linear.weight, 2))

    optimizer_model_1.zero_grad()
    loss_1.sum().backward()
    optimizer_model_1.step()

def calculate_weight_loss2(model, vnet, x, y):
    inputs, targets = x, y
    outputs = model(inputs).squeeze(-1)
    outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
    cost = nn.BCELoss(reduction='none')
    cost = cost(outputs, targets)  #
    cost_v = torch.reshape(cost, (len(cost), 1))

    with torch.no_grad():
        w_new = vnet(cost_v)
        w_new = abs((w_new - w_new.max()) / (w_new.max() - w_new.min()))

    return (cost_v * w_new), w_new

def print_eva(y_test, y_test_pred, y_test_pred_proba, item):
    data = {'y_test': y_test, 'y_evl': y_test_pred, 'y_pred_proba': y_test_pred_proba}
    res = pd.DataFrame(data)
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    sen, PPV, spe, NPV, f1_score, acc, precision, fpr, gmean = Eva_Matrix(res['y_test'], res['y_evl'])
    auc = roc_auc_score(res['y_test'], res['y_pred_proba'])
    auc = round(auc, 3)

    return auc, acc, sen, spe,  gmean, f1_score

seed_torch()
warnings.filterwarnings("ignore")
torch.manual_seed(123)

nums = 1
lr = 0.05
lmbda = args.lmbda
lmbda2 = args.lmbda2
num_epochs = args.num_epochs
batch_size = args.batch_size
batch_size_meta = args.batch_size_meta
OutPath = args.OutPath
proj1 = args.proj1
proj2 = args.proj2

model_path = OutPath + proj1 + '_vs_' + proj2
if not os.path.exists(model_path):
    os.mkdir(model_path)

for num in range(nums):
    x1, x1_test, y1, y1_test = load_data(args.DataPath, args.proj, args.proj1, args.proj2, args.splRat, args.repeat)
    print(x1.shape)
    print(x1_test.shape)
    print(Counter(y1))
    print(Counter(y1_test))

    sampleNum = x1.shape[0]
    featureDim = x1.shape[1]
    meta_data_size = int(sampleNum*0.1) + 1

    meta_x1, meta_y1 = extract_samples(x1, y1, meta_data_size)

    vnet_1 = VNet(1, 300, 1)
    model_1 = build_model(featureDim)

    criterion_1 = nn.BCELoss(reduction='none')
    optimizer_1 = optim.SGD(model_1.params(), lr=1e-3)

    criterion_vnet_1 = nn.BCELoss(reduction='none')
    optimizer_vnet_1 = torch.optim.Adam(vnet_1.parameters(), 1e-2)

    loss = nn.BCELoss(reduction='none')

    start = 0
    best_prec1 = 0
    best_epoch = 0

    W = torch.normal(0, 0.01, size=(featureDim, 1), requires_grad=True)

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        for batch_indices in data_iter(batch_size, x1, y1):
            x1_subset = x1[batch_indices]
            y1_subset = y1[batch_indices]
            y1_subset = torch.FloatTensor(y1_subset).squeeze(-1)
            meta_x1_subset, meta_y1_subset = next(data_iter_meta(batch_size_meta, meta_x1, meta_y1))

            train_auto_meta3(model_1, 
                            vnet_1, 
                            optimizer_1, 
                            optimizer_vnet_1, 
                            x1_subset, y1_subset,
                            meta_x1_subset, meta_y1_subset, 
                            epoch, lmbda, lmbda2)

            ###
            yhat = sigmoid_net(x1_subset)
            yhat = yhat.squeeze(-1)
            l = loss(yhat, y1_subset) + lmbda * F.l1_loss(W, target=torch.zeros_like(W.detach()),
                                                        size_average=False)

            l.sum().backward()
            updater(x1_subset.shape[0])

            if start == 0:
                start = 1
            else:
                with torch.no_grad():
                    sign_w = torch.sign(W)
                    model_1.linear.weight[torch.where(torch.abs(model_1.linear.weight) <= lmbda * 0.05)] = 0

        pkl_fils = os.path.join(model_path, f'epoch_{epoch}.pkl')
        with open(pkl_fils, 'wb') as pickle_file:
            pickle.dump((model_1, optimizer_1, vnet_1, optimizer_vnet_1), pickle_file)

        ## test set
        beta = model_1.linear.weight.detach().squeeze()
        with torch.no_grad():
            y_test_pred = norY(model_1(x1_test).squeeze(-1).detach().numpy()) 
            auc, acc, sen, spe,  gmean, f1_score = print_eva(y1_test, y_test_pred, model_1(x1_test).squeeze(-1).detach().numpy(), 'test')
        
        ## best perf
        if auc > best_prec1:
            betab = beta
            aucb, accb, senb, speb,  gmeanb, f1_scoreb  = auc, acc, sen, spe,  gmean, f1_score 
            best_epoch = epoch

    print('[{0}]\t {1:.3f}\t {2:.3f}\t {3:.3f}\t {4:.3f}\t {5:.3f}\t {6:.3f}\t'.
        format(best_epoch, aucb, accb, senb, speb,  gmeanb, f1_scoreb))

    sorted_indices = torch.argsort(torch.abs(betab), descending=True)
    non_zero_indices = torch.nonzero(betab)[:, 0]
    zero_indices = torch.nonzero(torch.eq(betab, 0))[:, 0]
    sorted_weights = torch.gather(betab, 0, sorted_indices)
    non_zero_weights = torch.gather(betab, 0, non_zero_indices)
    zero_weights = torch.gather(betab, 0, zero_indices)
    # print(betab)
    print(len(non_zero_weights))
    # print(non_zero_indices)
