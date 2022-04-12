from __future__ import print_function, division
import copy
import torch
import argparse
import numpy as np
import os
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score,\
f1_score, roc_auc_score, roc_curve, accuracy_score, matthews_corrcoef as mcc
from torch.utils.data import Subset
import pandas as pd

import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score, confusion_matrix
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve,\
davies_bouldin_score as dbs, normalized_mutual_info_score as nmi, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from scipy.stats import ttest_ind
from sklearn import model_selection, metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.cluster import KMeans
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import umap
import sys

import numbers
from sklearn.metrics import davies_bouldin_score as dbs, adjusted_rand_score as ari
from matplotlib import pyplot as plt
color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

from models import NNClassifier
from utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default= 'creditcard')
parser.add_argument('--input_dim', default= '-1')

# Training parameters
parser.add_argument('--lr', default= 0.002, type=float)
parser.add_argument('--alpha', default= 1, type=float)
parser.add_argument('--wd', default= 5e-4, type=float)
parser.add_argument('--batch_size', default= 512, type=int)
parser.add_argument('--n_epochs', default= 10, type=int)
parser.add_argument('--n_runs', default= 5, type=int)
parser.add_argument('--pre_epoch', default= 40, type=int)
parser.add_argument('--pretrain', default= True, type=bool)
parser.add_argument("--load_ae",  default=False, type=bool)
parser.add_argument("--classifier", default="LR")
parser.add_argument("--tol", default=0.01, type=float)
parser.add_argument("--attention", default="True")
parser.add_argument('--ablation', default='None')
parser.add_argument('--cluster_balance', default='hellinger')

# Model parameters
parser.add_argument('--lamda', default= 1, type=float)
parser.add_argument('--beta', default= 0.5, type=float) # KM loss wt
parser.add_argument('--gamma', default= 1.0, type=float) # Classification loss wt
parser.add_argument('--delta', default= 0.01, type=float) # Class seploss wt
parser.add_argument('--eta', default= 0.01, type=float) # Class seploss wt
parser.add_argument('--hidden_dims', default= [64, 32])
parser.add_argument('--n_z', default= 20, type=int)
parser.add_argument('--n_clusters', default= 3, type=int)
parser.add_argument('--clustering', default= 'cac')
parser.add_argument('--n_classes', default= 2, type=int)

# Utility parameters
parser.add_argument('--device', default= 'cpu')
parser.add_argument('--log_interval', default= 10, type=int)
parser.add_argument('--verbose', default= 'False')
parser.add_argument('--other', default= 'False')
parser.add_argument('--cluster_analysis', default= 'False')
parser.add_argument('--pretrain_path', default= '/Users/shivin/Document/NUS/Research/CAC/CAC_DL/DeepCAC/pretrained_model')


parser = parser.parse_args()
args = parameters(parser)

classifiers = ["LR", "SVM", "LDA", "RF", "KNN", "SGD", "Ridge", "MLP"]

test_results = pd.DataFrame(columns=['Dataset', 'Classifier', 'alpha',\
    'Base_F1_mean', 'Base_AUC_mean', 'Base_F1_std', 'Base_AUC_std',\
    'KM_F1_mean', 'KM_AUC_mean', 'KM_F1_std', 'KM_AUC_std',\
    'CAC_F1_mean', 'CAC_AUC_mean', 'CAC_F1_std', 'CAC_AUC_std'], dtype=object)

def get_classifier(classifier):
    if classifier == "LR":
        model = LogisticRegression(random_state=0, max_iter=1000)
    elif classifier == "RF":
        model = RandomForestClassifier(n_estimators=10)
    elif classifier == "SVM":
        # model = SVC(kernel="linear", probability=True)
        model = LinearSVC(max_iter=5000)
        model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
    elif classifier == "Perceptron":
        model = Perceptron(max_iter=1000)
        model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
    elif classifier == "ADB":
        model = AdaBoostClassifier(n_estimators = 100)
    elif classifier == "DT":
        model = DecisionTreeClassifier()
    elif classifier == "LDA":
        model = LDA()
    elif classifier == "NB":
        model = MultinomialNB()
    elif classifier == "SGD":
        model = SGDClassifier(loss='log', max_iter=1000)
    elif classifier == "Ridge":
        model = RidgeClassifier()
        model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
    elif classifier == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif classifier == "MLP":
        model = MLPClassifier(alpha=1e-3,
                     hidden_layer_sizes=(16, 8), random_state=108, max_iter=1000)
    else:
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
    return model


####################################################################################
####################################################################################
####################################################################################
###################################### Training ####################################
####################################################################################
####################################################################################
####################################################################################

f1_scores, auc_scores, acc_scores, auprc_scores = [], [], [], []

for classifier in classifiers:
    f1_scores, auc_scores, acc_scores = [], [], []
    for r in range(args.n_runs):
        scale, column_names, train_data, val_data, test_data = get_train_val_test_loaders(args, r_state=r)
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        train_loader = generate_data_loaders(X_train, y_train, args.batch_size)
        val_loader = generate_data_loaders(X_val, y_val, args.batch_size)
        test_loader = generate_data_loaders(X_test, y_test, args.batch_size)

        clf = get_classifier(classifier)
        # X_train = scale.fit_transform(X_train)
        # X_test = scale.fit_transform(X_test)
        clf.fit(X_train, y_train.ravel())
        preds = clf.predict(X_test)
        pred_proba = clf.predict_proba(X_test)
        f1_scores.append(f1_score(preds, y_test))
        auc_scores.append(roc_auc_score(y_test.ravel(), pred_proba[:,1]))
        auprc_scores.append(average_precision_score(y_test.ravel(), pred_proba[:,1]))
        acc_scores.append(accuracy_score(preds, y_test))

    print("Dataset\tCLF\tF1\tAUC\tAUPRC\tACC")
    print("{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format\
        (args.dataset, classifier, np.average(f1_scores), np.average(auc_scores), np.average(auprc_scores) ,np.average(acc_scores)))

print("\n\n")