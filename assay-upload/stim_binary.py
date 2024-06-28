# %%
import os
import random
import copy
import torch
from torch import nn
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.functional import F
# from net import Net1, VNet
from net import *
import argparse
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.model_selection import train_test_split

from net import *
from utils import *
from utils2 import *

np.random.seed(0)


# %% 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_features', type=int, default=1000, help='num_features')
parser.add_argument('--num_examples', type=int, default=200, help='num_examples')
parser.add_argument('--set_seed', type=int, default=42, help='set_seed')
parser.add_argument('--zero_ratio', type=float, default=0, help='zero_ratio')
parser.add_argument('--sigma', type=float, default=0.1, help='sigma')
parser.add_argument('--IR', type=float, default=2, help='imbalance ratio')
parser.add_argument('--repeat', type=int, default=1, help='repeat time')
parser.add_argument('--lmbda', type=float, default=0.05, help='lmbda')
parser.add_argument('--lmbda2', type=float, default=0.1, help='lmbda')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
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

class Lasso(nn.Module):
    def __init__(self, input_size):
        super(Lasso, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out


def soft_th(X, threshold=5):
    return np.sign(X) * np.maximum((np.abs(X) - threshold), np.zeros(X.shape))


def torch_soft_operator(X, threshold=0.1):
    np_X = X.numpy()
    tmp = np.sign(np_X) * np.maximum((np.abs(np_X) - threshold), np.zeros(np_X.shape))
    return torch.from_numpy(tmp.astype(np.float32))


def lasso(x, y, lmbda=1, lr=0.005, max_iter=2000, tol=1e-4, opt='SGD'):
    lso = Lasso(x.shape[1])
    criterion = nn.MSELoss(reduction='mean')
    if opt == 'adam':
        optimizer = optim.Adam(lso.parameters(), lr=lr)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(lso.parameters(), lr=lr)
    elif opt == "SGD":
        optimizer = optim.SGD(lso.parameters(), lr=lr)
    w_prev = torch.tensor(0.)
    for it in range(max_iter):
        # lso.linear.zero_grad()
        optimizer.zero_grad()
        out = lso(x)
        fn = criterion(out, y)
        l1_norm = lmbda * torch.norm(lso.linear.weight, p=1)
        l1_crit = nn.L1Loss()
        target = Variable(torch.from_numpy(np.zeros((x.shape[1], 1))))

        loss = 0.5 * fn + lmbda * F.l1_loss(lso.linear.weight, target=torch.zeros_like(lso.linear.weight.detach()),
                                            size_average=False)
        loss.backward()
        optimizer.step()
        # pdb.set_trace()
        if it == 0:
            w = lso.linear.weight.detach()
        else:
            with torch.no_grad():
                sign_w = torch.sign(w)
                ## hard-threshold
                lso.linear.weight[torch.where(torch.abs(lso.linear.weight) <= lmbda * lr)] = 0
                w = copy.deepcopy(lso.linear.weight.detach())
        if it % 500 == 0:
            #             print(target.shape)
            print(loss.item(), end=" ")
            print(torch.norm(lso.linear.weight.detach(), p=1), end=" ")
            print("l1_crit: ", l1_crit(lso.linear.weight.detach(), target), end=" ")
            print("F L1: ", F.l1_loss(lso.linear.weight.detach(), target=torch.zeros_like(lso.linear.weight.detach()),
                                      size_average=False))
    return lso.linear.weight.detach()


def norY(y):
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    return y


def sampleSelect(sample, x, y):
    pos = torch.where(sample == 3)
    neg = torch.where(sample == 0)
    X = x
    y_pos = y[pos]
    print(y_pos)
    y_neg = y[neg]
    y = torch.cat([y_pos, y_neg])
    print(y)
    # pos = pos.tolist(pos)
    pos_1 = [aa.tolist() for aa in pos]
    neg_1 = [aa.tolist() for aa in neg]
    pos_1 = pos_1[0]
    x_pos = x[pos_1, :]
    neg_1 = neg_1[0]
    x_neg = x[neg_1, :]
    x = torch.cat([x_pos, x_neg])
    return x, y


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield batch_indices


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


def accuracy(y_hat, y):  # @save
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) / len(y)


def accuracy2(y_hat, y):  # @save
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    else:
        y_hat = norY(y_hat)
    cmp = y_hat.type(y.dtype) == y
    acc_rate = float(cmp.type(y.dtype).sum()) / len(y)
    return acc_rate

def synthetic_binary_confident_data(w, num_seeds, num_examples, avg, std, sigma):  # @save
    x = np.random.normal(avg, std, (num_seeds, len(w)))
    x = torch.tensor(x)
    y = torch.matmul(x.to(torch.float32), w)
    y += sigma*torch.normal(0, 0.5, y.shape)
    y = torch.sigmoid(y)
    y = np.array(y)
    indx = 0
    ratio_high = 90
    ratio_medium = 60
    indx_high = np.empty(0)
    indx_medium = np.empty(0)
    indx_low = np.empty(0)
    y_orgrin = np.array(y, copy=True)
    ans = np.argsort(y_orgrin)
    y_orgrin = y_orgrin[ans]
    x = x[ans, :]
    y_mark = np.zeros((len(y), 1))
    for value_y in y_orgrin:
        if value_y >= np.percentile(y_orgrin, ratio_high) or value_y <= np.percentile(y_orgrin, 100 - ratio_high):
            indx_high = np.append(indx_high, indx)
            y_mark[indx] = 1
            y[indx] = 1 if value_y >= 0.5 else 0
        elif value_y >= np.percentile(y_orgrin, ratio_medium) or value_y <= np.percentile(y_orgrin, 100 - ratio_medium):
            indx_medium = np.append(indx_medium, indx)
            y[indx] = 1 if value_y >= 0.5 else 0
            y_mark[indx] = 2
        else:
            indx_low = np.append(indx_low, indx)
            y[indx] = 1 if value_y >= 0.5 else 0
            y_mark[indx] = 3
        indx += 1
    ratio_high = int(num_examples*0.1)
    criterion = np.zeros(num_seeds, dtype=bool)
    criterion[0:ratio_high], criterion[-ratio_high:] = True, True
    criterion[ratio_high+100:ratio_high+100+ratio_high], criterion[-(ratio_high+100+ratio_high):-(ratio_high+100)] = True, True

    M = int(num_examples * 0.3)
    C = num_seeds // 2
    if y[C] == 1:
        criterion[C:C + M] = True
        criterion[C-150:C-150 + M] = True
    else:
        criterion[C - M:C] = True
        criterion[C+150:C+150 + M] = True

    x = x[criterion, :]
    y = y[criterion]
    y_orgrin = y_orgrin[criterion]
    y_mark = y_mark[criterion]
    return x.to(torch.float32), y, y_orgrin, y_mark


class Accumulator:  # @save

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):  # @save
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def sgd(params, lr, batch_size):  # @save
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def updater(batch_size):
    return sgd([W], lr, batch_size)


def data_iter_meta(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

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
    cost = cost(outputs, targets) 
    cost_v = torch.reshape(cost, (len(cost), 1))

    with torch.no_grad():
        w_new = vnet(cost_v)
        w_new = abs((w_new - w_new.max()) / (w_new.max() - w_new.min()))

    return (cost_v * w_new), w_new


def synthetic_binary_data(w, num_features, num_examples, zero_ratio, sigma, IR, set_seed):
    true_w = w

    np.random.seed(set_seed)
    
    x = np.random.normal(0, 1, (2000, num_features))  # 生成特征数据
    x = torch.tensor(x)

    y = torch.matmul(x.to(torch.float32), true_w.detach())
    y += sigma * torch.normal(0, 0.5, y.shape)  # 添加高斯噪声
    y = torch.sigmoid(y)
    y = np.array(y)

    # 将预测结果转换为类别标签
    y = np.where(y >= 0.5, 1, 0)

    # 0类和1类样本
    num_ones_need = int(num_examples / (IR + 1))
    num_zeros_need = num_examples - num_ones_need

    one_indices = np.where(y == 1)[0]
    selected_one_indices = np.random.choice(one_indices, num_ones_need, replace=False)
    zero_indices = np.where(y == 0)[0]
    selected_zero_indices = np.random.choice(zero_indices, num_zeros_need, replace=False)

    selected_indices = np.concatenate((selected_one_indices, selected_zero_indices))
    X_train = x[selected_indices]
    y_train = y[selected_indices]

    return X_train.to(torch.float32), y_train

def print_eva(y_test, y_test_pred, y_test_pred_proba, item):
    data = {'y_test': y_test, 'y_evl': y_test_pred, 'y_pred_proba': y_test_pred_proba}
    res = pd.DataFrame(data)
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    sen, PPV, spe, NPV, f1_score, acc, precision, fpr, gmean = Eva_Matrix(res['y_test'], res['y_evl'])
    auc = roc_auc_score(res['y_test'], res['y_pred_proba'])
    auc = round(auc, 3)

    return auc, acc, sen, spe,  gmean, f1_score

lr = 0.05
nums = 1

lmbda = args.lmbda
lmbda2 = args.lmbda2
num_epochs = args.num_epochs
batch_size = args.batch_size
batch_size_meta = args.batch_size_meta

featureDim = args.num_features
sampleNum = args.num_examples
zero_ratio = args.zero_ratio
sigma = args.sigma
ir = args.IR
set_seed = args.set_seed

for num in range(nums):
    best_prec1 = 0
    best_epoch = 0

    true_w_1 = torch.zeros(featureDim)
    true_w_1[0:9] = 1
    true_w_1[10:15] = -1
    true_w_1[16:19] = 2

    x1, y1 = synthetic_binary_data(true_w_1, featureDim, sampleNum, 0, sigma, ir, set_seed)
    x1_test, y1_test = synthetic_binary_data(true_w_1, featureDim, 50, 0, sigma, 1, set_seed)


    W = torch.normal(0, 0.01, size=(featureDim, 1), requires_grad=True)

    meta_data_size = int(sampleNum*0.1) + 1
    meta_x1 = np.concatenate((x1[0:meta_data_size], x1[sampleNum-meta_data_size:sampleNum]), axis=0)
    meta_x1 = torch.tensor(meta_x1)
    meta_y1 = np.concatenate((y1[0:meta_data_size], y1[sampleNum-meta_data_size:sampleNum]), axis=0)

    vnet_1 = VNet(1, 300, 1)
    model_1 = build_model(featureDim)

    criterion_1 = nn.BCELoss(reduction='none')
    optimizer_1 = optim.SGD(model_1.params(), lr=1e-3)

    criterion_vnet_1 = nn.BCELoss(reduction='none')
    optimizer_vnet_1 = torch.optim.Adam(vnet_1.parameters(), 1e-2)

    loss = nn.BCELoss(reduction='none')

    batch_size = 10
    batch_size_meta = 10
    start = 0
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
                    model_1.linear.weight[torch.where(torch.abs(model_1.linear.weight) <= lmbda * lr)] = 0

        ## test set
        beta = model_1.linear.weight.detach().squeeze()


        # print(beta)
        with torch.no_grad():
            y_test_pred = norY(model_1(x1_test).squeeze(-1).detach().numpy()) 
            auc, acc, sen, spe,  gmean, f1_score = print_eva(y1_test, y_test_pred, model_1(x1_test).squeeze(-1).detach().numpy(), 'test')
            # print('[{0}]\t {1:.3f}\t {2:.3f}\t {3:.3f}\t {4:.3f}\t {5:.3f}\t {6:.3f}\t'.
            #     format(epoch, auc, acc, sen, spe,  gmean, f1_score))

            outputs = model_1(x1).squeeze(-1)
            outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
            cost = nn.BCELoss(reduction='none')
            y1_2 = torch.FloatTensor(y1).squeeze(-1)
            cost = cost(outputs, torch.tensor(y1_2)) 
            cost_v = torch.reshape(cost, (len(cost), 1)).detach()
            v_weght = vnet_1(cost_v)
            v_weght_new = abs((v_weght - v_weght.max()) / (v_weght.max() - v_weght.min()))
            merged_tensor = torch.cat((cost_v, v_weght_new), dim=1)
            numpy_array = merged_tensor.numpy()
            w_v_df = pd.DataFrame(numpy_array, columns=['loss', 'weight_v'])

        ## save model
        if auc > best_prec1:
            betab = beta
            w_v_dfb = w_v_df
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

    abs_tensor = torch.abs(betab)
    top_values, top_indices = torch.topk(abs_tensor, k=20)

    print(np.sum(top_indices.numpy() < 20))
    print(len(non_zero_weights))

    # out_dir = '../result_upload/'
    # pref = 'n' + str(sampleNum) + '_IR' + str(ir) + '_sigma' + str(sigma)
    # w_v_dfb.to_csv(out_dir + pref + '.csv', index=False)
