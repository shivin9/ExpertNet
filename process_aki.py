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
DATASET = '/Users/shivin/Document/NUS/Research/Data/aki48_new/'
ORIGINAL = os.path.abspath(".")

data_train = get_aki_static(DATASET+'train', ORIGINAL)
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

data_test = get_aki_static(DATASET+ 'test', ORIGINAL)
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

	
X = get_aki_static(DATASET+'train', ORIGINAL)
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
X_test = get_aki_static(DATASET+'test', ORIGINAL)
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

