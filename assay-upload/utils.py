import os
import random
import copy
import torch
from torch import nn
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.functional import F
from sklearn.metrics import confusion_matrix
from statistics import mean
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, f1_score

def seed_torch(seed=0):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

##=============== Data process ============

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # random.seed(42)
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        if len(batch_indices) < batch_size:
            continue
        yield batch_indices

def data_iter_meta(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # random.seed(42)
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        if len(batch_indices) < batch_size:
            continue
        yield features[batch_indices], labels[batch_indices]

def load_data(DataPath, proj, proj1, proj2, splRat, repeat):

    filePath = DataPath + proj + '_' + proj1 + '_vs_' + proj2 + '_merge_full_data.csv'
    MergeData = pd.read_csv(filePath, index_col=0)
    MergeData = MergeData.reset_index(drop=True)
    MergeData.head()

    ## Train-Test Data Splitting
    array = MergeData.values
    X = array[:, 1:MergeData.shape[1]] # features
    y = MergeData.Group.values #y = array[:, 0]
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # print('Number of samples:', n_samples)
    # print('Number of features:', n_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size = 1-splRat, 
                        stratify = y,
                        random_state = repeat)

    # print('X_train shape:', X_train.shape)
    # print('X_test shape:', X_test.shape)
    # print('y_train shape:', y_train.shape)
    # print('y_test shape:', y_test.shape)

    ## Data standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    # y_train = y_train.float()
    # y_test = y_test.float()
    return X_train, X_test, y_train, y_test

def load_data_df(DataPath, proj, proj1, proj2, splRat, repeat):

    filePath = DataPath + proj + '_' + proj1 + '_vs_' + proj2 + '_merge_full_data.csv'
    MergeData = pd.read_csv(filePath, index_col=0)

    X = MergeData.drop(columns=['Group'])
    y = MergeData['Group']
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size = 1-splRat, 
                        stratify = y,
                        random_state = repeat)

    return X_train, X_test, y_train, y_test

def load_data_multi(DataPath, proj, splRat, repeat):

    # DataPath = args.DataPath
    # splRat = args.splRat
    # repeat = args.repeat

    filePath = DataPath + proj + '_multiple_merge_full_data.csv'
    MergeData = pd.read_csv(filePath, index_col=0)
    MergeData = MergeData.reset_index(drop=True)
    # MergeData.head()

    ## Train-Test Data Splitting
    array = MergeData.values
    X = array[:, 1:MergeData.shape[1]] # features
    y = MergeData.Group.values #y = array[:, 0]
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # print('Number of samples:', n_samples)
    # print('Number of features:', n_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size = 1-splRat, 
                        stratify = y,
                        random_state = repeat)

    # print('X_train shape:', X_train.shape)
    # print('X_test shape:', X_test.shape)
    # print('y_train shape:', y_train.shape)
    # print('y_test shape:', y_test.shape)

    ## Data standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    # y_train = y_train.float()
    # y_test = y_test.float()
    return X_train, X_test, y_train, y_test    

def extract_samples(x, y, n):
    unique_labels = np.unique(y) 
    selected_indices = []
    for label in unique_labels:
        indices = np.where(y == label)[0]
        selected_indices.extend(np.random.choice(indices, n, replace=False))
    selected_x = x[selected_indices]
    selected_y = y[selected_indices]
    return selected_x, selected_y

##=============== Simulated data  ============
def synthetic_binary_data(num_features, num_examples, zero_ratio, sigma, IR, set_seed):
    
    # 生成随机的真实权重向量
    # true_w = np.random.normal(0, 1, num_features).astype(np.float32)  

    # 固定的权重
    # true_w = torch.zeros(num_features)
    # true_w[0:99] = 1
    # true_w[100:199] = -1
    # true_w[200:399] = 0
    # true_w[400:599] = 2

    # true_w = torch.zeros(num_features)
    # true_w[0:9] = 1
    # true_w[10:15] = -1
    # true_w[16:19] = 2

    true_w = torch.zeros(num_features)
    true_w[0:9] = 1
    true_w[10:15] = -1
    true_w[16:19] = 2
    true_w = true_w * 3

    np.random.seed(set_seed)
    
    # 将一部分权重设为0
    # num_zero_coef = int(zero_ratio * num_features)  # 计算需要设为0的权重个数
    # zero_indices = np.random.choice(num_features, num_zero_coef, replace=False)
    # true_w[zero_indices] = 0

    x = np.random.normal(0, 1, (2000, num_features))  # 生成特征数据
    x = torch.tensor(x)

    # y = torch.matmul(x.to(torch.float32), torch.tensor(true_w))  # 计算线性模型的预测结果
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
    x = x[selected_indices]
    y = y[selected_indices]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return X_train.to(torch.float32), X_test.to(torch.float32), y_train, y_test, true_w

##=============== Performance Evaluation ============
def norY(y):
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    return y

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def validate(batch_size, x1_test, y1_test, model, criterion):
    top1 = AverageMeter()
    sens = AverageMeter()
    spes = AverageMeter()
    gmeans = AverageMeter()
    # losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for batch_indices in data_iter(batch_size, x1_test, y1_test):

        x1_subset_test = x1_test[batch_indices]
        y1_subset_test = y1_test[batch_indices]
        y1_subset_test = torch.FloatTensor(y1_subset_test).squeeze(-1)

        with torch.no_grad():
            yhat = model(x1_subset_test)
            yhat = yhat.squeeze(-1)
        
        prec1 = accuracy2(yhat, y1_subset_test)
        sen, spe, gmean = g_mean(yhat, y1_subset_test)

        top1.update(prec1, x1_test.size(0))
        sens.update(sen, x1_test.size(0))
        spes.update(spe, x1_test.size(0))
        gmeans.update(gmean, x1_test.size(0))

        # loss = criterion(yhat, y1_subset_test)
        # losses.update(loss, x1_test.size(0))

    # print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # return prec1, sen, spe, gmean
    return top1.avg, sens.avg, spes.avg, gmeans.avg

def accuracy(y_hat, y): 
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    acc = np.mean(y_hat.numpy() == y)
    return float(acc)

def accuracy2(y_hat, y):  # @save
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    else:
        y_hat = norY(y_hat)
    cmp = y_hat.type(y.dtype) == y
    acc_rate = float(cmp.type(y.dtype).sum()) / len(y)
    return acc_rate

def g_mean(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    else:
        y_hat = norY(y_hat) 
    true_positives = ((y_hat == 1) & (y == 1)).sum()
    true_negatives = ((y_hat == 0) & (y == 0)).sum()
    actual_positives = (y == 1).sum()
    actual_negatives = (y == 0).sum()
    sensitivity = true_positives / actual_positives
    specificity = true_negatives / actual_negatives
    g_mean = (sensitivity * specificity) ** 0.5
    return sensitivity, specificity, g_mean

def metric_test(yhat, y1_test):
    yhat2 = yhat.detach().numpy()
    if len(yhat2.shape) > 1 and yhat2.shape[1] > 1:
        yhat2 = yhat2.argmax(axis=1)
    else:
        yhat2 = norY(yhat2) 
    correct = (yhat2 == y1_test).sum().item()
    total = y1_test.shape[0]
    acc = correct / total
    true_positives = ((yhat2 == 1) & (y1_test == 1)).sum()
    true_negatives = ((yhat2 == 0) & (y1_test == 0)).sum()
    actual_positives = (y1_test == 1).sum()
    actual_negatives = (y1_test == 0).sum()
    sensitivity = true_positives / actual_positives
    specificity = true_negatives / actual_negatives
    g_mean = (sensitivity * specificity) ** 0.5
    # auc = roc_auc_score(y1_test, yhat2)
    auc = roc_auc_score(y1_test, yhat) 
    precision = precision_score(y1_test, yhat2)
    recall = sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall)
    return auc, acc, sensitivity, specificity, g_mean, f1_score

def Print_para(print_cont, model_1):
    print(print_cont)
    model_dict = model_1.state_dict()
    for key, value in model_dict.items():
        # print(key)
        for i, v in enumerate(value[:5]):
            print(f"Value {i+1}: {v}")