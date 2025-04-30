import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance

from utils import *

def Compare_main_pip_cv(x_train, y_train, x_test, y_test, model_meth):
    if isinstance(x_train, torch.Tensor): 
        x_train = x_train.numpy()
    elif isinstance(x_train, np.ndarray): 
        pass
    else:
        raise TypeError("Neither x_train is Tensor nor ndarray")

    if model_meth == "elasticnet":
        clf = LogisticRegression(random_state = 123, penalty='elasticnet', solver='saga', multi_class='auto')
        param_grid = {'C': [0.1, 1, 10, 50],
                        'max_iter': [100, 300, 500],
                        'l1_ratio': [0.1, 0.5, 0.9]  
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='accuracy')
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        coefficients_abs = np.abs(model.coef_[0])
        sorted_indices = np.argsort(coefficients_abs)[::-1]
        top_features = sorted_indices[:20]

    if model_meth == 'elasticnetCS':
        clf = LogisticRegression(random_state=123, 
                         penalty='elasticnet', 
                         class_weight='balanced', 
                         solver='saga', 
                         multi_class='auto')
        param_grid = {'C': [0.1, 1, 10, 50], 
                      'max_iter': [100, 300, 500], 
                      'l1_ratio': [0.1, 0.5, 0.9]
                      }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='accuracy')
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        coefficients_abs = np.abs(model.coef_[0])
        sorted_indices = np.argsort(coefficients_abs)[::-1]
        top_features = sorted_indices[:20]

    if model_meth == "RF":
        clf = RandomForestClassifier()
        param_grid = {'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 100, None],
                    'max_features': sp_randint(1, 11),
                    'min_samples_split': sp_randint(1, 11),
                    'min_samples_leaf': sp_randint(1, 11),
                    'bootstrap': [True, False],
                    'criterion': ['gini', 'entropy']}

        cv = StratifiedKFold(n_splits= 5, shuffle = True, random_state= 123)
        gs = RandomizedSearchCV(clf, param_distributions = param_grid, n_iter = 1,  cv = cv, scoring= 'accuracy')
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_features = sorted_indices[:20]

    if model_meth == 'SVM':
        clf = svm.SVC(probability=True)
        param_grid = {'C': [1, 10, 100], 
                      'gamma': [0.1, 10, 100], 
                      'kernel':('linear', 'rbf')}
        gs = GridSearchCV(clf, param_grid, cv=5)
        gs.fit(x_train, y_train)
        model = svm.SVC(**gs.best_params_, probability=True)
        model.fit(x_train, y_train)
        r = permutation_importance(model, x_train, y_train, n_repeats=20, random_state=123, n_jobs = -1)
        imp = r.importances_mean
        imp_abs = np.abs(imp)
        sorted_indices = np.argsort(imp_abs)[::-1]
        top_features = sorted_indices[:20]

    if model_meth == 'KNN':
        clf = KNeighborsClassifier()
        param_grid = {'n_neighbors':[1,2, 3,4, 5,7]}
        gs = GridSearchCV(clf, param_grid, cv = 5)
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        r = permutation_importance(model, x_train, y_train, n_repeats=20, random_state=123, n_jobs = -1)
        imp = r.importances_mean
        imp_abs = np.abs(imp)
        sorted_indices = np.argsort(imp_abs)[::-1]
        top_features = sorted_indices[:20]
        # print(model.get_params())
    
    if model_meth == "XGBoost":
        clf = xgb.XGBClassifier(random_state=123, objective='binary:logistic', eval_metric='logloss')
        param_grid = {'n_estimators': [50, 100, 150],  # Number of boosting rounds
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size for each boosting round
                        'max_depth': [3, 5, 7],  # Maximum depth of trees
                        'subsample': [0.8, 0.9, 1.0],  # Fraction of samples to use for each boosting round
                        'colsample_bytree': [0.8, 0.9, 1.0]  # Fraction of features to use for each tree
                        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='accuracy')
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_features = sorted_indices[:20]

    if model_meth == "AdaBoost":
        base_estimator = DecisionTreeClassifier(max_depth=1)
        clf = AdaBoostClassifier(base_estimator=base_estimator, random_state=123)
        param_grid = {'n_estimators': [50, 100, 150],  # Number of boosting rounds
                      'learning_rate': [0.01, 0.05, 0.1, 0.2]  # Step size for each boosting round
                      }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='accuracy')
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_features = sorted_indices[:20]

    if model_meth == "LightGBM":
        clf = lgb.LGBMClassifier(random_state=123)
        param_grid = {
            'n_estimators': [50, 100, 150],  # Number of boosting rounds
            'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size for each boosting round
            'max_depth': [3, 5, 7],  # Maximum depth of trees
            'num_leaves': [31, 50, 100],  # Number of leaves in full trees
            'subsample': [0.8, 0.9, 1.0]  # Fraction of samples to use for each boosting round
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='accuracy')
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_features = sorted_indices[:20]

    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_train_pred_proba = model.predict_proba(x_train)
    y_test_pred = model.predict(x_test)
    y_test_pred_proba = model.predict_proba(x_test)

    if(y_test_pred_proba.shape[1]) > 1:
        auc, acc, sen, spe, gmean, f1_score, AUPRC, MCC, balanced_accuracy = print_eva(y_test, y_test_pred, y_test_pred_proba[:,1], 'test')
    else:
        auc, acc, sen, spe, gmean, f1_score, AUPRC, MCC, balanced_accuracy = print_eva(y_test, y_test_pred, y_test_pred_proba, 'test')
    
    return auc, acc, sen, spe, gmean, f1_score, AUPRC, MCC, balanced_accuracy

    # print('AUC: {:.3f}\t ACC: {:.3f}\t SEN: {:.3f}\t SPE:{:.3f}\t Gmean:{:.3f}\t f1score:{:.3f}\t AUPRC:{:.3f}\t MCC:{:.3f}\t balanced_accuracy:{:.3f}\t'.
    #       format(auc, acc, sen, spe, gmean, f1_score, AUPRC, MCC, balanced_accuracy))

def Compare_main_pip_cv_multi(x_train, y_train, x_test, y_test, model_meth):
    if isinstance(x_train, torch.Tensor): 
        x_train = x_train.numpy()
    elif isinstance(x_train, np.ndarray): 
        pass
    else:
        raise TypeError("Neither x_train is Tensor nor ndarray")

    if model_meth == "elasticnet":
        clf = LogisticRegression(random_state = 123, penalty='elasticnet', solver='saga', multi_class='auto')
        param_grid = {'C': [0.01, 0.1, 1],
                        'max_iter': [100, 200, 300],
                        'l1_ratio': [0.1, 0.5, 0.9]  
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='accuracy')
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        coefficients_abs = np.abs(model.coef_[0])
        sorted_indices = np.argsort(coefficients_abs)[::-1]
        top_features = sorted_indices[:20]

    if model_meth == 'elasticnetCS':
        clf = LogisticRegression(random_state=123, 
                         penalty='elasticnet', 
                         class_weight='balanced', 
                         solver='saga', 
                         multi_class='auto')
        param_grid = {'C': [0.1, 1, 10], 
                      'max_iter': [100, 200, 300], 
                      'l1_ratio': [0.1, 0.5, 0.9]
                      }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='accuracy')
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        coefficients_abs = np.abs(model.coef_[0])
        sorted_indices = np.argsort(coefficients_abs)[::-1]
        top_features = sorted_indices[:20]

    if model_meth == "RF":
        clf = RandomForestClassifier()
        param_grid = {'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 100, None],
                    'max_features': sp_randint(1, 11),
                    'min_samples_split': sp_randint(1, 11),
                    'min_samples_leaf': sp_randint(1, 11),
                    'bootstrap': [True, False],
                    'criterion': ['gini', 'entropy']}

        cv = StratifiedKFold(n_splits=5, shuffle = True, random_state= 123)
        gs = RandomizedSearchCV(clf, param_distributions = param_grid, n_iter = 1,  cv = cv, scoring= 'accuracy')
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_features = sorted_indices[:20]

    if model_meth == 'SVM':
        clf = svm.SVC(probability=True)
        param_grid = {'C': [1, 10, 50], 
                      'gamma': [0.1, 10, 30], 
                      'kernel':('linear', 'rbf')}
        gs = GridSearchCV(clf, param_grid, cv=5)
        gs.fit(x_train, y_train)
        model = svm.SVC(**gs.best_params_, probability=True)
        model.fit(x_train, y_train)
        r = permutation_importance(model, x_train, y_train, n_repeats=20, random_state=123, n_jobs = -1)
        imp = r.importances_mean
        imp_abs = np.abs(imp)
        sorted_indices = np.argsort(imp_abs)[::-1]
        top_features = sorted_indices[:20]

    if model_meth == 'KNN':
        clf = KNeighborsClassifier()
        param_grid = {'n_neighbors':[1,2, 3,4, 5,7]}
        gs = GridSearchCV(clf, param_grid, cv = 5)
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        r = permutation_importance(model, x_train, y_train, n_repeats=20, random_state=123, n_jobs = -1)
        imp = r.importances_mean
        imp_abs = np.abs(imp)
        sorted_indices = np.argsort(imp_abs)[::-1]
        top_features = sorted_indices[:20]
        # print(model.get_params())
    
    if model_meth == "XGBoost":
        clf = xgb.XGBClassifier(random_state=123, objective='binary:logistic', eval_metric='logloss')
        param_grid = {'n_estimators': [50, 100, 150],  # Number of boosting rounds
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size for each boosting round
                        'max_depth': [3, 5, 7],  # Maximum depth of trees
                        'subsample': [0.8, 0.9, 1.0],  # Fraction of samples to use for each boosting round
                        'colsample_bytree': [0.8, 0.9, 1.0]  # Fraction of features to use for each tree
                        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='accuracy')
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_features = sorted_indices[:20]

    if model_meth == "AdaBoost":
        base_estimator = DecisionTreeClassifier(max_depth=1)
        clf = AdaBoostClassifier(base_estimator=base_estimator, random_state=123)
        param_grid = {'n_estimators': [50, 100, 150],  # Number of boosting rounds
                      'learning_rate': [0.01, 0.05, 0.1, 0.2]  # Step size for each boosting round
                      }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='accuracy')
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_features = sorted_indices[:20]

    if model_meth == "LightGBM":
        clf = lgb.LGBMClassifier(random_state=123)
        param_grid = {
            'n_estimators': [50, 100, 150],  # Number of boosting rounds
            'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size for each boosting round
            'max_depth': [3, 5, 7],  # Maximum depth of trees
            'num_leaves': [31, 50, 100],  # Number of leaves in full trees
            'subsample': [0.8, 0.9, 1.0]  # Fraction of samples to use for each boosting round
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='accuracy')
        gs.fit(x_train, y_train)
        model = gs.best_estimator_
        model.fit(x_train, y_train)
        feature_importances = model.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_features = sorted_indices[:20]

    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_train_pred_proba = model.predict_proba(x_train)
    y_test_pred = model.predict(x_test)
    y_test_pred_proba = model.predict_proba(x_test)

    df_train = OVR_eval_df(y_train, y_train_pred, y_train_pred_proba)
    df_test = OVR_eval_df(y_test, y_test_pred, y_test_pred_proba)
    
    return df_train, df_test, model