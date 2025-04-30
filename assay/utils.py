import os
import random
import copy
import torch
from torch import nn
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.functional import F
from statistics import mean
import numpy as np
import pandas as pd
import os

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, scale, Normalizer
from sklearn.metrics import roc_auc_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, StratifiedKFold
from sklearn import metrics
import sklearn.metrics as mx
# from sklearn.metrics import roc_curve, plot_roc_curve
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, matthews_corrcoef, average_precision_score
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report, RocCurveDisplay,auc

from re import A
import numpy as np
from numpy import mean, std
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from collections import Counter, defaultdict
import mplcursors

##=============== Pre-work ============
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print(path + ' creat successfully')
        return True
    else:
        print(path + ' existed')
        return False

def seed_torch(seed=0):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

##=============== Simulated data  ============
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

def synthetic_binary_data(w, num_features, num_examples, zero_ratio, sigma, IR, set_seed):
    true_w = w

    np.random.seed(set_seed)
    
    # 将一部分权重设为0
    # num_zero_coef = int(zero_ratio * num_features) 
    # zero_indices = np.random.choice(num_features, num_zero_coef, replace=False)
    # true_w[zero_indices] = 0

    x = np.random.normal(0, 1, (2000, num_features)) 
    x = torch.tensor(x)

    # y = torch.matmul(x.to(torch.float32), torch.tensor(true_w))
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

##=============== Data process ============
# 批量数据生成器，用于生成训练数据的批次
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

# 批量数据生成器（带返回），用于生成元学习数据的批次
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

def data_iter_meta_class_bal(batch_size, features, labels):
    features = features.clone().detach()
    labels = torch.tensor(labels)
    num_examples = len(features)
    
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label.item()].append(idx)

    all_classes = list(class_indices.keys())

    while True:
        batch_features = []
        batch_labels = []
        
        for cls in all_classes:
            indices = class_indices[cls]
            random.shuffle(indices)  
            num_samples =  int(batch_size // 2)
            
            batch_indices = torch.tensor(indices[:num_samples])
            batch_features.append(features[batch_indices])
            batch_labels.append(labels[batch_indices])

        if len(batch_features) == len(all_classes):
            batch_features = torch.cat(batch_features, dim=0)
            batch_labels = torch.cat(batch_labels, dim=0).numpy() 
            yield batch_features, batch_labels
        else:
            break

# 数据加载函数，加载并处理数据 
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

def load_data_val_2(DataPath, proj, proj1, proj2,repeat, splRat=0.8,folds_num=5):
    # 构建数据文件路径
    filePath = DataPath + proj + '_' + proj1 + '_vs_' + proj2 + '_merge_full_data.csv'
    # 读取数据
    MergeData = pd.read_csv(filePath, index_col=0)
    MergeData = MergeData.reset_index(drop=True)
    MergeData.head()

    # 训练集和测试集划分
    array = MergeData.values
    X = array[:, 1:MergeData.shape[1]] # 特征
    y = MergeData.Group.values # 标签

    # 创建第一次划分：80% 作为训练+验证，20% 作为测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-splRat, random_state=repeat, stratify=y)

    # 数据标准化处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # print("Training set shape:", X_train.shape)
    # print("Training set distribution:", Counter(y_train))
    # print("Testing set shape:", X_test.shape)
    # print("Testing set distribution:", Counter(y_test))

    # 第二步：五折交叉验证，确保每个fold中少数样本均衡分布
    skf = StratifiedKFold(n_splits=folds_num)
    folds = []

    # 循环生成每个折
    for fold_index, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        
        # 将每个折的训练和验证数据添加到列表
        folds.append({
            'train': (X_fold_train, y_fold_train),
            'val': (X_fold_val, y_fold_val)
        })
        # print(f"Fold {fold_index + 1}:")
        # print("X_fold_train set shape:", X_fold_train.shape)
        # print("y_fold_train set distribution:", Counter(y_fold_train))
        # print("X_fold_val set shape:", X_fold_val.shape)
        # print("y_fold_val set distribution:", Counter(y_fold_val))

    return folds, X_train, y_train, X_test, y_test

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

# 样本提取函数，用于从数据中抽取特定数量的样本
def extract_samples(x, y, n):
    unique_labels = np.unique(y) 
    selected_indices = []
    for label in unique_labels:
        indices = np.where(y == label)[0]
        selected_indices.extend(np.random.choice(indices, n, replace=False))
    selected_x = x[selected_indices]
    selected_y = y[selected_indices]
    return selected_x, selected_y

# 将数据转化为Variable类型
def to_var(x, requires_grad=True):
    return Variable(x, requires_grad=requires_grad)

# 归一化函数，将预测值映射到0和1之间
def norY(y):
    y[y >= 0.5] = 1
    y[y < 0.5] = 0
    return y

##=============== Model evaluation ============
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

def Print_para(model_1):
    model_dict = model_1.state_dict()
    for key, value in model_dict.items():
        # print(key)
        for i, v in enumerate(value[:5]):
            print(f"Value {i+1}: {v}")

# 针对二分类的评估函数
def Eva_Matrix(Y_val, Ypred_SVM):
    cmat = confusion_matrix(Y_val, Ypred_SVM)
    TN = cmat[0][0]
    TP = cmat[1][1]
    FP = cmat[0][1]
    FN = cmat[1][0]

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    sensitivity = TP / (TP + FN)
    precision = TP / (TP + FP)
    PPV = TP / (TP + FP)

    specificity = TN / (TN + FP)
    NPV = TN / (TN + FN)
    
    fpr = FP / (FP + TN)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    Gmeans = math.sqrt(sensitivity * specificity )

    accuracy = round(accuracy, 3)
    sensitivity = round(sensitivity, 3)
    specificity = round(specificity, 3)
    fpr = round(fpr, 3)
    precision = round(precision, 3)
    f1_score = round(f1_score, 3)
    PPV = round(PPV, 3)
    NPV = round(NPV, 3)
    Gmeans = round(Gmeans, 3)

    # Calculate AUPRC
    precision_vals, recall_vals, _ = precision_recall_curve(Y_val, Ypred_SVM)
    AUPRC = auc(recall_vals, precision_vals)  # AUC for the Precision-Recall curve
    AUPRC = round(AUPRC, 3)

    # Calculate MCC (Matthews Correlation Coefficient)
    MCC = matthews_corrcoef(Y_val, Ypred_SVM)
    MCC = round(MCC, 3)

   # Calculate Balanced Accuracy
    balanced_accuracy = balanced_accuracy_score(Y_val, Ypred_SVM)
    balanced_accuracy = round(balanced_accuracy, 3)

    return sensitivity, PPV, specificity, NPV, f1_score, accuracy, precision, fpr, Gmeans, AUPRC, MCC, balanced_accuracy

def print_eva(y1_test, y_test_pred, y_test_pred_proba, item):
    data = {'y_test': y1_test, 'y_evl': y_test_pred, 'y_pred_proba': y_test_pred_proba}
    res = pd.DataFrame(data)
    sen, PPV, spe, NPV, f1_score, acc, precision, fpr, gmean, AUPRC, MCC, balanced_accuracy = Eva_Matrix(res['y_test'], res['y_evl'])
    auc = roc_auc_score(res['y_test'], res['y_pred_proba'])
    auc = round(auc, 3)
    return auc, acc, sen, spe,  gmean, f1_score, AUPRC, MCC, balanced_accuracy

def print_ROC(y_true, y_pred_proba, pic_path):

    y_true = y_true.tolist()
    y_pred_proba = y_pred_proba[:,1].tolist()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    youden_index = tpr - fpr
    best_threshold = thresholds[youden_index.argmax()]
    best_fpr = fpr[youden_index.argmax()]
    best_tpr = tpr[youden_index.argmax()]
    auc_value = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label= ' AUC = {:.2f}'.format(auc_value))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (area = 0.5)')
    plt.scatter(best_fpr, best_tpr, c='red', marker='o', label=f'Youden Index Max ({best_threshold:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')

    plt.savefig(pic_path, dpi = 600)
    plt.show()
    plt.close()

# 针对多分类的评估函数
def calculate_performance(y_true, y_pred, y_pred_proba, class_label):
    y_true_binary = [1 if label == class_label else 0 for label in y_true]
    y_pred_binary = [1 if label == class_label else 0 for label in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    g_mean = (sensitivity * specificity) ** 0.5
    acc = accuracy_score(y_true_binary, y_pred_binary)
    # auc = roc_auc_score(y_true_binary, y_pred_binary)
    ovr_prob = y_pred_proba[:, class_label] 
    auc  = roc_auc_score(y_true_binary, ovr_prob)
    auprc = average_precision_score(y_true_binary, ovr_prob)

    f1 = f1_score(y_true_binary, y_pred_binary)
    mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
    balanced_acc = balanced_accuracy_score(y_true_binary, y_pred_binary)
    return sensitivity, specificity, g_mean, acc, f1, auc, auprc, mcc, balanced_acc

def OVR_eval_df(y_test, y_test_pred, y_test_pred_proba):
    specificity_arr = []
    sensitivity_arr = []
    g_mean_arr = []
    auc_arr = []
    acc_arr = []
    auprc_arr = []
    f1_score_arr = []
    mcc_arr = []
    balanced_acc_arr = []
    # auc = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='macro')
    for class_label in set(y_test):
        sensitivity, specificity, g_mean, acc, f1, auc, auprc, mcc, balanced_acc = calculate_performance(y_test, y_test_pred, y_test_pred_proba, class_label)
        specificity_arr.append(specificity)
        sensitivity_arr.append(sensitivity)
        g_mean_arr.append(g_mean)
        auc_arr.append(auc)
        auprc_arr.append(auprc)
        acc_arr.append(acc)
        f1_score_arr.append(f1)
        mcc_arr.append(mcc)
        balanced_acc_arr.append(balanced_acc)

    df = pd.DataFrame({'auc': auc_arr, 
                       'acc': acc_arr,
                       'sen': sensitivity_arr, 
                       'spe': specificity_arr,
                       'gmean': g_mean_arr, 
                       'f1_score': f1_score_arr,
                       'auprc': auprc_arr, 
                       'mcc': mcc_arr,  
                       'balanced_accuracy': balanced_acc_arr },
    )
    return df

def evaluate_beta_performance(pred_beta, true_beta):

    coefidx = [i for i, coef in enumerate(pred_beta) if coef != 0]
    betaidx = [i for i, beta in enumerate(true_beta) if beta != 0]

    coeftrans = [0] * len(pred_beta)
    betatrans = [0] * len(true_beta)

    for idx in coefidx:
        coeftrans[idx] = 1
    for idx in betaidx:
        betatrans[idx] = 1

    TN = sum([(1 - coeftrans[i]) * (1 - betatrans[i]) for i in range(len(pred_beta))])
    FP = sum([coeftrans[i] * (1 - betatrans[i]) for i in range(len(pred_beta))])
    FN = sum([(1 - coeftrans[i]) * betatrans[i] for i in range(len(pred_beta))])
    TP = sum([coeftrans[i] * betatrans[i] for i in range(len(pred_beta))])

    accuracy = (TP + TN) / (TN + TP + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return accuracy, sensitivity, specificity

def VnetPanting(vnet_1):
    # 创建一个包含0到1000的输入数组
    inputs = torch.arange(0, 100).unsqueeze(1)
    inputs = inputs.float()

    # 将输入传递给vnet_1()并获取结果
    results = vnet_1(inputs)
    results = results.detach().numpy()

    # 绘制图表
    fig, ax = plt.subplots()
    ax.plot(inputs, results)
    cursor = mplcursors.cursor(ax, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        sel.annotation.set_text(f"({x:.1f}, {y:.1f})")

    plt.xlabel('Loss')
    plt.ylabel('Sample weight')
    plt.title('vnet_1 Results')
    plt.show()

def VnetPanting_multi(vnet_1):
    # 创建一个包含0到1000的输入数组
    inputs = torch.arange(0, 10).unsqueeze(1)
    inputs = inputs.float()

    # 将输入传递给vnet_1()并获取结果
    results = vnet_1(inputs)
    results = results.detach().numpy()

    # 绘制图表
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))  # 3行1列子图
    for i in range(3):
        ax = axes[i]
        ax.plot(inputs, results[:, i]) 
        ax.set_xlabel('Loss')
        ax.set_ylabel('Sample weight')
        ax.set_title(f'vnet_1 Results - Column {i+1}')
        
        cursor = mplcursors.cursor(ax, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            sel.annotation.set_text(f"({x:.1f}, {y:.1f})")

    plt.tight_layout()
    plt.show()

##=============== Model selection  ============
def divide_and_adjust(numerator, denominator):
    if denominator == 0:
        return "Division by zero is not allowed"  # 处理除以零的情况
    
    # 进行整除运算，判断余数
    quotient = numerator // denominator
    remainder = numerator % denominator
    
    if remainder != 0:
        return quotient + 1  # 若有余数则加1
    
    return quotient  # 否则直接返回商

def model_select(model1, model2, x_val, y_val):
    y_val_pred_1 = norY(model1(x_val).squeeze(-1).detach().numpy())
    y_val_pred_2 = norY(model2(x_val).squeeze(-1).detach().numpy())
    
    best_model = 0
    auc1 = roc_auc_score(y_val, y_val_pred_1)
    auc2 = roc_auc_score(y_val, y_val_pred_2)
    # print('model1 auc:', auc1)
    # print('model2 auc:', auc2)
    if auc1 < auc2:
        best_model = 1
    return best_model

def model_evaluate(model, VNet, x_test, y_test, vnet_trend):
    with torch.no_grad():
        y_test_pred = norY(model(x_test).squeeze(-1).detach().numpy()) 
        auc, acc, sen, spe,  gmean, f1_score, AUPRC, MCC, balanced_accuracy = print_eva(y_test, y_test_pred, model(x_test).squeeze(-1).detach().numpy(), 'test')
    vnet_trend = vnet_trend

    # print('ACC: {:.3f}\t SEN: {:.3f}\t SPE:{:.3f}\t Gmean:{:.3f}\t f1score:{:.3f}\t AUPRC:{:.3f}\t MCC:{:.3f}\t balanced_accuracy:{:.3f}\t'.
    # format(auc, acc, sen, spe, gmean, f1_score, AUPRC, MCC, balanced_accuracy))
    
    return auc, acc, sen, spe,  gmean, f1_score, AUPRC, MCC, balanced_accuracy
