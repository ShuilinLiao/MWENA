# %% 
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from net import *
from utils import *
from utils2 import *
import warnings
import pickle

# %% 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--DataPath', type=str, default='../result_upload/', help = 'Path of input data')
parser.add_argument('--OutPath', type=str, default='../result_upload/', help='Output directory')
parser.add_argument('--item', type=str, default='b', help='binary (b) or multiple (m)')
parser.add_argument('--proj', type=str, default='exoRBase', help='exoRBase, PDAC or ILD')
parser.add_argument('--proj1', type=str, default='BRCA', help='proj1')
parser.add_argument('--proj2', type=str, default='OV', help='proj2')
parser.add_argument('--splRat', type=float, default=0.8, help='split ratio')
parser.add_argument('--repeat', type=int, default=1, help='repeat time')
parser.add_argument('--num_epochs', type=str, default='49', help='number of epochs')
args = parser.parse_args()

# %% 
epoch = args.num_epochs
OutPath = args.OutPath
proj = args.proj
proj1 = args.proj1
proj2 = args.proj2

model_pre = OutPath + proj1 + '_vs_' + proj2
x1, x1_test, y1, y1_test = load_data(args.DataPath, proj, args.proj1, args.proj2, args.splRat, args.repeat)
filePath = args.DataPath + proj + '_' + proj1 + '_vs_' + proj2 + '_merge_full_data.csv'
pkl_fils = os.path.join(model_pre, f'Epoch_{epoch}.pkl')

print(x1.shape)
print(x1_test.shape)
print(Counter(y1))
print(Counter(y1_test))

MergeData = pd.read_csv(filePath, index_col=0)
all_genes = MergeData.columns[1:].tolist()

with open(pkl_fils, 'rb') as pickle_file:
    loaded_data = pickle.load(pickle_file)
model_1 = loaded_data[0]
optimizer_1 = loaded_data[1]
vnet_1 = loaded_data[2]
optimizer_vnet_1 = loaded_data[3]

def print_eva(y1_test, y_test_pred, y_test_pred_proba, item):
    data = {'y_test': y1_test, 'y_evl': y_test_pred, 'y_pred_proba': y_test_pred_proba}
    res = pd.DataFrame(data)
    sen, PPV, spe, NPV, f1_score, acc, precision, fpr, gmean = Eva_Matrix(res['y_test'], res['y_evl'])
    auc = roc_auc_score(res['y_test'], res['y_pred_proba'])
    auc = round(auc, 3)

    return auc, acc, sen, spe,  gmean, f1_score

print('train: ')
with torch.no_grad():
    # yhat = model_1(x1).squeeze(-1)
    # y_train_pred = norY(yhat)
    y_test_pred = norY(model_1(x1).squeeze(-1).detach().numpy()) 
    auc, acc, sen, spe,  gmean, f1_score = print_eva(y1, y_test_pred, model_1(x1).squeeze(-1).detach().numpy(), 'train')

print('[{0}]\t {1:.3f}\t {2:.3f}\t {3:.3f}\t {4:.3f}\t {5:.3f}\t {6:.3f}\t'.
    format(epoch, auc, acc, sen, spe,  gmean, f1_score))

print('test: ')
with torch.no_grad():
    # yhat = model_1(x1_test).squeeze(-1)
    # y_test_pred = norY(yhat)
    y_test_pred = norY(model_1(x1_test).squeeze(-1).detach().numpy()) 
    auc, acc, sen, spe,  gmean, f1_score = print_eva(y1_test, y_test_pred, model_1(x1_test).squeeze(-1).detach().numpy(), 'test')

print('[{0}]\t {1:.3f}\t {2:.3f}\t {3:.3f}\t {4:.3f}\t {5:.3f}\t {6:.3f}\t'.
    format(epoch, auc, acc, sen, spe,  gmean, f1_score))

# %%
beta = model_1.linear.weight.detach().squeeze()
sorted_indices = torch.argsort(torch.abs(beta), descending=True)
non_zero_indices = torch.nonzero(beta)[:, 0]
zero_indices = torch.nonzero(torch.eq(beta, 0))[:, 0]

sorted_genes = [all_genes[i.item()] for i in sorted_indices]
non_zero_genes = [all_genes[i.item()] for i in non_zero_indices]
zero_genes = [all_genes[i.item()] for i in zero_indices]

sorted_weights = torch.gather(beta, 0, sorted_indices)
non_zero_weights = torch.gather(beta, 0, non_zero_indices)
zero_weights = torch.gather(beta, 0, zero_indices)

print(sorted_genes[:30])
print(sorted_weights[:30])

print(len(non_zero_weights))
# %%
