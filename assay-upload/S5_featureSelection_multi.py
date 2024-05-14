# %% 
from collections import Counter
import numpy as np

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
parser.add_argument('--item', type=str, default='m', help='binary (b) or multiple (m)')
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

model_pre = OutPath + proj + '_multiple' 
x1, x1_test, y1, y1_test = load_data_multi(args.DataPath, proj, args.splRat, args.repeat)
filePath = args.DataPath + proj + '_multiple_merge_full_data.csv'
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

print('train: ')
with torch.no_grad():
    yhat = model_1(x1).squeeze(-1)
    y_train_pred = yhat.argmax(axis=1)
    
df_test = OVR_eval_df(y1, y_train_pred, yhat)
print(df_test)

print('test: ')
with torch.no_grad():
    yhat = model_1(x1_test).squeeze(-1)
    y_test_pred = yhat.argmax(axis=1)
    
df_test = OVR_eval_df(y1_test, y_test_pred, yhat)
print(df_test)

# %%
beta = model_1.linear.weight.detach()
first_beta = beta[0]

def get_weights(beta):
    sorted_indices = torch.argsort(torch.abs(beta), descending=True)
    non_zero_indices = torch.nonzero(beta)[:, 0]
    zero_indices = torch.nonzero(torch.eq(beta, 0))[:, 0]

    sorted_genes = [all_genes[i.item()] for i in sorted_indices]
    non_zero_genes = [all_genes[i.item()] for i in non_zero_indices]
    zero_genes = [all_genes[i.item()] for i in zero_indices]

    sorted_weights = torch.gather(beta, 0, sorted_indices)
    non_zero_weights = torch.gather(beta, 0, non_zero_indices)
    zero_weights = torch.gather(beta, 0, zero_indices)
    
    return sorted_indices, non_zero_indices, zero_indices, \
           sorted_genes, non_zero_genes, zero_genes, \
           sorted_weights, non_zero_weights, zero_weights

sorted_indices1, non_zero_indices1, zero_indices1, \
        sorted_genes1, non_zero_genes1, zero_genes1, \
        sorted_weights1, non_zero_weights1, zero_weights1 = get_weights(beta[0])

sorted_indices2, non_zero_indices2, zero_indices2, \
        sorted_genes2, non_zero_genes2, zero_genes2, \
        sorted_weights2, non_zero_weights2, zero_weights2 = get_weights(beta[1])

sorted_indices3, non_zero_indices3, zero_indices3, \
        sorted_genes3, non_zero_genes3, zero_genes3, \
        sorted_weights3, non_zero_weights3, zero_weights3 = get_weights(beta[2])

# average_beta = torch.mean(torch.stack([beta[0], beta[1], beta[2]]), dim=0)
average_beta = (torch.abs(beta[0]) + torch.abs(beta[1]) + torch.abs(beta[2])) / 3 
sorted_indices_aver, non_zero_indices_aver, zero_indices_aver, \
        sorted_genes_aver, non_zero_genes_aver, zero_genes_aver, \
        sorted_weights_aver, non_zero_weights_aver, zero_weights_aver = get_weights(average_beta)

# print(sorted_genes1[:30])
# print(sorted_weights1[:30])

# print(sorted_genes2[:30])
# print(sorted_weights2[:30])

# print(sorted_genes3[:30])
# print(sorted_weights3[:30])

print(sorted_genes_aver[:30])
print(sorted_weights_aver[:30])

sorted_indices = torch.argsort(torch.abs(average_beta), descending=True)
non_zero_indices = torch.nonzero(average_beta)[:, 0]
zero_indices = torch.nonzero(torch.eq(average_beta, 0))[:, 0]
sorted_weights = torch.gather(average_beta, 0, sorted_indices)
non_zero_weights = torch.gather(average_beta, 0, non_zero_indices)
zero_weights = torch.gather(average_beta, 0, zero_indices)
print(len(non_zero_weights))
