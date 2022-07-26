from sklearn.preprocessing import * 
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from read_patients import *
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Ridge
# from fancyimpute import KNN
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
# from training_aki import *
import pandas as pd
import os

NUM_CLUSTERS = 4
DATASET = '/Users/shivin/Document/NUS/Research/Data/aki/'
ORIGINAL = os.path.abspath(".")

data_train = get_aki(DATASET+'train', ORIGINAL)
X = pd.concat(data_train,axis=1).T

# X['Sofa_O2'] = X.apply(lambda x: Sofa_Oxygen(x.SaO2, x.FiO2), axis=1)
# X['Sofa_MAP'] = X['MAP'].apply(Sofa_MAP)
# X['Sofa_Bilirubin'] = X['Bilirubin_total'].apply(Sofa_Bilirubin)
# X['Sofa_Creatinin'] = X['Creatinine'].apply(Sofa_Creatinine)
# X['Sofa_Platelets'] = X['Platelets'].apply(Sofa_Platelets)

columns = X.columns

data_columns = list(columns[1:90]) + ['y'] # get the columns which have data, not mask
non_binary_columns = data_columns[:81] # only these columns have non-binary data fit for scaling

X = X.fillna(0)
X = X[data_columns]

y_train = X['y']
X_train = X.drop(columns=['y'])

scaler = MinMaxScaler()
X_train[non_binary_columns] = scaler.fit_transform(np.nan_to_num(X[non_binary_columns]))

# FS, ESTIMATORS, alt_clusterings, X_val, y_val = cluster_case_2(data, NUM_CLUSTERS, NUM_CLUSTERS, USE_FULL_FEATURES, CLUSTERINGS)

data_test = get_aki(DATASET+ 'test', ORIGINAL)
X_test = pd.concat(data_test,axis=1).T
columns = X_test.columns

data_columns = list(columns[1:90]) + ['y'] # get the columns which have data, not mask
non_binary_columns = data_columns[:81] # only these columns have non-binary data fit for scaling

X_test = X_test.fillna(0)
X_test = X_test[data_columns]
y_test = X_test['y']
X_test = X_test.drop(columns=['y'])

X_test[non_binary_columns] = scaler.transform(np.nan_to_num(X_test[non_binary_columns]))

# scaler = MinMaxScaler()
X_total = X_train.append(X_test)
y_total = y_train.append(y_test)
y_total = y_total.astype(int)

y_total = y_total.astype(int)

X_total.to_csv(DATASET+"X.csv", index=False)
y_total.to_csv(DATASET+"y.csv", index=False)

train_case_1(X, X_test)
train_case_2(X, X_test, ENSEMBLE=True)

################################################################
######################### ARDS Dataset #########################
################################################################

NUM_CLUSTERS = 4
DATASET = '/Users/shivin/Document/NUS/Research/Data/ards_new/'


X = get_aki(DATASET+'train', ORIGINAL)
X = pd.concat(X,axis=1).T
columns = X.columns

data_columns = list(columns[1:90]) + list(columns[172:]) # get the columns which have data, not mask
non_binary_columns = data_columns[:81] # only these columns have non-binary data fit for scaling

X = X.fillna(0)
X = X[data_columns]

y_train = X['y']
X_train = X.drop(columns=['y'])


scaler = MinMaxScaler()
X_train[non_binary_columns] = scaler.fit_transform(np.nan_to_num(X_train[non_binary_columns]))

#### Test
X_test = get_aki(DATASET+'test', ORIGINAL)
X_test = pd.concat(X_test,axis=1).T
columns = X_test.columns

data_columns = list(columns[1:90]) + list(columns[172:]) # get the columns which have data, not mask
non_binary_columns = data_columns[:81] # only these columns have non-binary data fit for scaling

X_test = X_test.fillna(0)
X_test = X_test[data_columns]

y_test = X_test['y']
X_test = X_test.drop(columns=['y'])

# scaler = MinMaxScaler()
X_test[non_binary_columns] = scaler.transform(np.nan_to_num(X_test[non_binary_columns]))

X_total = X_train.append(X_test)
y_total = y_train.append(y_test)
y_total = y_total.astype(int)

X_total.to_csv(DATASET+"X.csv", index=False)
y_total.to_csv(DATASET+"y.csv", index=False)

train_case_1(X, X_test)
train_case_2(X, X_test, ENSEMBLE=True)


################################################################
######################## ARDS TS Dataset #######################
################################################################

final_train = get_aki_TS('./train', ori_direc=os.curdir, t_end=24)
final_test = get_aki_TS('./test', ori_direc=os.curdir, t_end=24)
y_train = []

for i in range(len(final_train)):
    final_train[i] = final_train[i][data_columns]
    final_train[i] = final_train[i].ffill(axis=0)
    y_train.append(final_train[i].y.iloc[-1])
    final_train[i] = final_train[i].drop(columns=['y'])

y_test = []

for i in range(len(final_test)):
    final_test[i] = final_test[i][data_columns]
    final_test[i] = final_test[i].ffill(axis=0)
    y_test.append(final_test[i].y.iloc[-1])
    final_test[i] = final_test[i].drop(columns=['y'])

X_tr = []
X_te = []

for i in range(len(final_train)):
	X_tr.append(final_train[i].to_numpy())

for i in range(len(final_test)):
	X_te.append(final_test[i].to_numpy())

trn_len = []
te_len = []

for i in range(len(final_train)):
	trn_len.append(len(final_train[i]))

for i in range(len(final_test)):
	te_len.append(len(final_test[i]))