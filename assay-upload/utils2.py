from re import A
import numpy as np
from numpy import mean, std
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn import metrics
import sklearn.metrics as mx
# from sklearn.metrics import roc_curve, plot_roc_curve
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report, RocCurveDisplay,auc
from sklearn.preprocessing import scale
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_curve, auc

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\") # 去除尾部 \ 符号
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print(path + ' creat successfully')
        return True
    else:
        print(path + ' existed')
        return False

##### =============== Data processing ==============

def data_process(full_data):

    Y=full_data.iloc[:,full_data.columns == 'Group']
    X=full_data.iloc[:,full_data.columns != 'Group']

    Y = np.ravel(Y)

    # Split the df into 80% train 20% test data; then further split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.20, random_state=1303, stratify=Y, shuffle = True)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size= 0.20, random_state=1303,stratify=Y_train,shuffle = True)

    # print(Y_train[(Y_train=='0')].count())
    # print(len(Y_train))
    # print(Y_val[(Y_val=='0')].count())
    # print(len(Y_val))
    # print(Y_test[(Y_test=='0')].count())
    # len(Y_test)

    X_tr=X_tr.replace(0.0, np.nan)
    perc = 80.0
    min_count =  int(((100-perc)/100)*X_tr.shape[0] + 1)
    Xtr_filtered = X_tr.dropna( axis=1, thresh=min_count)
    # X_tr.shape
    # Xtr_filtered.shape

    filtered_features= X_val.columns.intersection(Xtr_filtered.columns)
    Xval_filtered=X_val[filtered_features]
    # Xval_filtered.shape

    Xtr_filtered=Xtr_filtered.fillna(0)
    Xval_filtered=Xval_filtered.fillna(0)

    # Xtr_filtered.iloc[:,Xtr_filtered.columns.str.startswith('exo_circ')]

    transformer = Normalizer()
    Xtr_filtered_S = transformer.fit_transform(Xtr_filtered)
    Xval_filtered_S = transformer.transform(Xval_filtered)

    return Xtr_filtered, Xtr_filtered_S, Y_tr, Xval_filtered, Xval_filtered_S, Y_val,  X_test, Y_test

def data_process_validation(full_data):

    Y = full_data.iloc[:,full_data.columns == 'Group']
    X = full_data.iloc[:,full_data.columns != 'Group']
    Y = np.ravel(Y)

    X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size= 0.20, random_state=1303,stratify=Y,shuffle = True)

    transformer = Normalizer()
    X_tr_S = transformer.fit_transform(X_tr)
    X_tr_S = pd.DataFrame(X_tr_S, columns=X_tr.columns)

    X_te_S = transformer.transform(X_val)
    X_te_S = pd.DataFrame(X_te_S, columns=X_val.columns)

    return X_tr_S, Y_tr, X_te_S, Y_val

def data_process_short(full_data):

    Y = full_data.iloc[:,full_data.columns == 'Group']
    X = full_data.iloc[:,full_data.columns != 'Group']
    Y = np.ravel(Y)

    return X, Y

def data_process_short2(full_data):

    Y = full_data.iloc[:,full_data.columns == 'Group']
    X = full_data.iloc[:,full_data.columns != 'Group']
    Y = np.ravel(Y)

    transformer = Normalizer()
    X_tr_S = transformer.fit_transform(X)
    X_tr_S = pd.DataFrame(X, columns=X.columns)

    return X_tr_S, Y
    
##### =============== Model evaluation ==============

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

    return sensitivity, PPV, specificity, NPV, f1_score, accuracy, precision, fpr, Gmeans

def get_auc(model_SVM, XvalSVM_S, Y_val):
    
    y_probabilities = model_SVM.predict_proba(XvalSVM_S)
    y_scores = y_probabilities[:, 1]
    auc = roc_auc_score(Y_val, y_scores)
    auc = round(auc, 3)

    return auc

def print_eva(y_test, y_test_pred, y_test_pred_proba, item):
    if(y_test_pred_proba.shape[1]) > 1:
        data = {'y_test': y_test, 'y_evl': y_test_pred, 'y_pred_proba': y_test_pred_proba[:,1]}
    else:
        data = {'y_test': y_test, 'y_evl': y_test_pred, 'y_pred_proba': y_test_pred_proba}
    res = pd.DataFrame(data)
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    sensitivity, PPV, specificity, NPV, f1_score, accuracy, precision, fpr, Gmeans = Eva_Matrix(res['y_test'], res['y_evl'])
    auc = roc_auc_score(res['y_test'], res['y_pred_proba'])
    auc = round(auc, 3)
    conf_res = confusion_matrix(res['y_test'], res['y_evl'])

    # met_res = np.array([auc, sensitivity, PPV, specificity, NPV, f1_score, accuracy, precision, fpr, Gmeans], dtype=np.float64)
    # met_dat = pd.DataFrame(met_res, columns=[item])
    # met_dat.index = ['auc', 'sensitivity', 'PPV', 'specificity', 'NPV', 'f1_score', 'accuracy', 'precision', 'fpr', 'Gmeans']

    met_res = np.array([auc, accuracy, sensitivity, specificity, Gmeans, f1_score], dtype=np.float64)
    met_dat = pd.DataFrame(met_res, columns=[item])
    met_dat.index = ['auc', 'acc', 'sen', 'spe', 'gmean', 'f1_score']

    return res, conf_res, met_dat

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
    f1 = f1_score(y_true_binary, y_pred_binary)
    return sensitivity, specificity, g_mean, acc, f1, auc

def OVR_eval_df(y_test, y_test_pred, y_test_pred_proba):
    specificity_arr = []
    sensitivity_arr = []
    g_mean_arr = []
    auc_arr = []
    acc_arr = []
    f1_score_arr = []
    # auc = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='macro')
    for class_label in set(y_test):
        sensitivity, specificity, g_mean, acc, f1, auc = calculate_performance(y_test, y_test_pred, y_test_pred_proba, class_label)
        specificity_arr.append(specificity)
        sensitivity_arr.append(sensitivity)
        g_mean_arr.append(g_mean)
        auc_arr.append(auc)
        acc_arr.append(acc)
        f1_score_arr.append(f1)

    df = pd.DataFrame({'AUC': auc_arr, 'ACC': acc_arr, 
    'sensitivity': sensitivity_arr, 'specificity': specificity_arr, 
    'g_mean_arr': g_mean_arr, 'f1_score': f1_score_arr})
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
