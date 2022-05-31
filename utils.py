## Utils.py
from __future__ import division, print_function
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import mutual_info_score, roc_auc_score, average_precision_score, accuracy_score, davies_bouldin_score as dbs
from sklearn import metrics
from sklearn.metrics.cluster import silhouette_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, label_binarize
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.stats import ttest_ind, wasserstein_distance as wd
from read_patients import get_aki
from matplotlib import pyplot as plt
import sys
import umap

color = ['grey', 'red', 'blue', 'pink', 'olive', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan']
DATASETS = ['diabetes', 'ards', 'ards_new', 'ihm', 'cic', 'cic_new', 'sepsis', 'aki', 'aki_new', 'infant', 'wid_mortality',\
            'synthetic', 'titanic', 'magic', 'adult', 'creditcard', 'heart', 'cic_los', 'cic_los_new', 'paper_synthetic']

DATA_DIR = "/Users/shivin/Document/NUS/Research/Data"
BASE_DIR = "/Users/shivin/Document/NUS/Research/cac/cac_dl/ExpertNet"

def avg(x):
    if len(x) == 0:
        return 0
    else:
        return np.average(x)

np.avg = avg

# Disable Print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore Print
def enablePrint():
    sys.stdout = sys.__stdout__


def calculate_bound(model, B, m):
    sum1 = 0
    for j in range(model.n_clusters):
        prod = 1.0
        for param in model.classifiers[j][0].parameters():
            prod *= torch.norm(param.view(-1))
        sum1 += B[j]*prod
    x = sum1/np.sqrt(model.n_clusters*m)
    return x.item()


def silhouette_new(X, labels, metric="euclidean"):
    if len(np.unique(labels)) == 1:
        return 0
    else:
        return silhouette_score(X, labels, metric=metric)
        # return dbs(X, labels)


def calculate_mutual_HTFD(X, cluster_ids):
    feature_diff = 0
    cntr = 0
    n_clusters = len(torch.unique(cluster_ids))
    n_columns = X.shape[1]
    top_quartile = np.int(n_columns/4)
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i > j:
                ci = torch.where(cluster_ids == i)[0]
                cj = torch.where(cluster_ids == j)[0]
                Xi = X[ci]
                Xj = X[cj]
                # feature_diff += sum(ttest_ind(Xi, Xj, axis=0)[1] < 0.05)/n_columns
                # Take the max element and take negative exp weighted by 0.05, top 5 features
                col_p_val = np.sort(np.nan_to_num(ttest_ind(Xi, Xj, axis=0, equal_var=True)[1]))[::-1]
                feature_diff += np.sum(np.exp(-col_p_val[:top_quartile]/0.05))/top_quartile
                # feature_diff += torch.nn.functional.kl_div(Xi.log(), Xj, reduction='batchmean')
                cntr += 1
    if cntr == 0:
        return 0
    return feature_diff/cntr


def calculate_HTFD(X_train, cluster_ids):
    print("\nCluster Wise discriminative features (HTFD)")
    cluster_entrpy = 0
    cntr = 0
    n_features = X_train.shape[1]
    n_clusters = len(torch.unique(cluster_ids))
    input_dim = X_train.shape[1]
    HTFD_scores = {}
    top_quartile = np.int(n_features/4)
    final_score = 0
    for i in range(n_clusters):
        HTFD_scores[i] = {}
        ci = torch.where(cluster_ids == i)[0]
        if len(ci) < 2:
            return 0
        # Collect features of all the columns
        for c in range(n_features):
            Xi_c = X_train[ci][:,c]
            Zc = []
            # Collect values from other clusters
            for j in range(n_clusters):
                if i != j:
                    cj = torch.where(cluster_ids == j)[0]
                    if len(X_train[cj].shape) == 1:
                        Xj_c = X_train[cj].reshape(1,n_features)[:,c]
                    else:
                        Xj_c = X_train[cj][:,c]
                    Zc = np.concatenate([Zc, Xj_c])

            col_entrpy = 0
            p_vals = np.nan_to_num(ttest_ind(Xi_c, Zc, axis=0, equal_var=True))[1]
            HTFD_scores[i][c] = np.round(-np.log(p_vals + np.finfo(float).eps)*0.05, 3)

        sorted_dict = sorted(HTFD_scores[i].items(), key=lambda item: item[1])[::-1]
        HTFD_cluster_score = 0
        for feature, p_val in sorted_dict:
            HTFD_cluster_score += p_val

        final_score += HTFD_cluster_score/n_features
    return final_score/n_clusters


def shannon_entropy(A, mode="auto", verbose=False):
     """
     https://stackoverflow.com/questions/42683287/python-numpy-shannon-entropy-array
     """
     A = np.asarray(A)
 
     # Determine distribution type
     if mode == "auto":
         condition = np.all(A.astype(float) == A.astype(int))
         if condition:
             mode = "discrete"
         else:
             mode = "continuous"
     if verbose:
         print(mode, file=sys.stderr)
     # Compute shannon entropy
     pA = A / A.sum()
     # Remove zeros
     pA = pA[np.nonzero(pA)[0]]
     if mode == "continuous":
         return -np.sum(pA*np.log2(A))
     if mode == "discrete":
         return -np.sum(pA*np.log2(pA))


def plot_hist(x, y, bins=10):
    minn = min(np.min(x), np.min(y))
    maxx = max(np.max(x), np.max(y))
    range_x = np.max(x) - np.min(x)
    range_y = np.max(y) - np.min(y)
    gap = min(range_x, range_y)/bins

    if gap < 1e-5:
        return 0

    h1, r1 = np.histogram(x, np.arange(minn-gap, maxx+gap, gap))
    h2, r2 = np.histogram(y, np.arange(minn-gap, maxx+gap, gap))
    k1 = [(r1[j]+r1[j-1])/2 for j in range(1, len(r1))]
    k2 = [(r2[j]+r2[j-1])/2 for j in range(1, len(r2))]
    # print(h1, h2, k1)

    fig, axes = plt.subplots(1, 2)

    axes[0].bar(k1, h1, label='X1', alpha=.5)
    axes[1].bar(k2, h2, label='X2', alpha=.5)
    # axes[0].hist(h1, bins=r1, label='X1', alpha=.5)
    # axes[1].hist(h2, bins=r2, label='X2', alpha=.5)
    plt.show()


def calc_MI(x, y, c=0, bins=10):
    minn = min(np.min(x), np.min(y))
    maxx = max(np.max(x), np.max(y))
    range_x = np.max(x) - np.min(x)
    range_y = np.max(y) - np.min(y)
    gap = min(range_x, range_y)/bins

    if gap < 1e-5:
        return 0

    h1 = np.histogram(x, np.arange(minn-gap, maxx+gap, gap))[0]
    h2 = np.histogram(y, np.arange(minn-gap, maxx+gap, gap))[0]
    c_xy = np.histogram2d(h1, h2, bins)[0]

    if np.sum(c_xy) == 0:
        return 0

    h_x = shannon_entropy(h1)
    h_y = shannon_entropy(h2)
    # https://stats.stackexchange.com/questions/468585/mutual-information-between-a-vector-and-a-constant
    if h_x * h_y == 0:
        # print("Column: ", c, x, y)
        # print(h1, h2, range_x, range_y, mutual_info_score(None, None, contingency=c_xy))
        return 0
    mi = mutual_info_score(None, None, contingency=c_xy)/np.sqrt(h_x * h_y)
    return mi


def calculate_MIFD(X, cluster_ids):
    cluster_entrpy = 0
    cntr = 0
    n_columns = X.shape[1]
    n_clusters = len(torch.unique(cluster_ids))
    top_quartile = np.int(n_columns/4)
    col_entrpy = np.zeros(n_columns)
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i > j:
                col_entrpy *= 0
                ci = torch.where(cluster_ids == i)[0]
                cj = torch.where(cluster_ids == j)[0]
                Xi = X[ci]
                Xj = X[cj]
                for c in range(n_columns):
                    col_entrpy[c] = calc_MI(Xi[:,c], Xj[:,c], c)
                # Sort col_entrpy
                col_entrpy = np.sort(col_entrpy)[::-1]
                cluster_entrpy += np.sum(col_entrpy[:top_quartile])/top_quartile
                cntr += 1
    if cntr == 0:
        return 0
    return cluster_entrpy/cntr


def calculate_WDFD(X, cluster_ids):
    cluster_entrpy = 0
    cntr = 0
    n_columns = X.shape[1]
    n_clusters = len(torch.unique(cluster_ids))
    top_quartile = np.int(n_columns/4)
    col_entrpy = np.zeros(n_columns)
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i > j:
                col_entrpy *= 0
                ci = torch.where(cluster_ids == i)[0]
                cj = torch.where(cluster_ids == j)[0]
                if len(ci) < 2 or len(cj) < 2:
                    return 0
                Xi = X[ci]
                Xj = X[cj]
                for c in range(n_columns):
                    col_entrpy[c] = wd(Xi[:,c], Xj[:,c])
                # Sort col_entrpy
                col_entrpy = np.sort(col_entrpy)[::-1]
                cluster_entrpy += np.sum(col_entrpy[:top_quartile])/top_quartile
                cntr += 1
    if cntr == 0:
        return 0
    return cluster_entrpy/cntr


def MIFD_Cluster_Analysis(X_train, cluster_ids, column_names):
    print("\nCluster Wise discriminative features (MIFD)")
    cluster_entrpy = 0
    cntr = 0
    n_columns = X_train.shape[1]
    n_clusters = len(torch.unique(cluster_ids))
    input_dim = X_train.shape[1]
    mi_scores = {}
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i > j:
                joint_col_name = str(i) + "," + str(j)
                mi_scores[joint_col_name] = {}
                ci = torch.where(cluster_ids == i)[0]
                cj = torch.where(cluster_ids == j)[0]
                Xi = X_train[ci]
                Xj = X_train[cj]
                col_entrpy = 0
                for c in range(n_columns):
                    c_entropy = calc_MI(Xi[:,c], Xj[:,c], 0)
                    col_entrpy += c_entropy
                    mi_scores[joint_col_name][c] = np.round(c_entropy, 3)
                    # print(column_names[c], ":", c_entropy)
                cluster_entrpy += col_entrpy/n_columns
                cntr += 1
                print("\n========\n")
                print(joint_col_name)
                sorted_dict = sorted(mi_scores[joint_col_name].items(), key=lambda item: -item[1])[:10]
                for k, v in sorted_dict:
                    c = column_names[k]
                    print("Feature:", c, "Val:", v, "C1:")
                    print(np.mean(Xi[:,k]), "C2:", np.mean(Xj[:,k]))


def HTFD_Cluster_Analysis(X_train, y_train, cluster_ids, column_names):
    print("\nCluster Wise discriminative features (HTFD)")
    cluster_entrpy = 0
    cntr = 0
    n_columns = X_train.shape[1]
    n_clusters = len(torch.unique(cluster_ids))
    print(n_clusters)
    input_dim = X_train.shape[1]
    mi_scores = {}
    for i in range(n_clusters):
        ci = torch.where(cluster_ids == i)[0]
        print("|C{}| = {}".format(i, np.bincount(y_train[ci])/len(ci)))
        for j in range(n_clusters):
            if i > j:
                joint_col_name = str(i) + "," + str(j)
                mi_scores[joint_col_name] = {}
                cj = torch.where(cluster_ids == j)[0]
                Xi = X_train[ci]
                Xj = X_train[cj]
                col_entrpy = 0
                p_vals = np.nan_to_num(ttest_ind(Xi, Xj, axis=0, equal_var=True))[1]
                for c in range(n_columns):
                    # mi_scores[joint_col_name][c] = np.round(p_vals[c], 3)
                    mi_scores[joint_col_name][c] = np.round(-np.log(p_vals[c] + np.finfo(float).eps)*0.05, 3)
                    # print(column_names[c], ":", c_entropy)
                cntr += 1
                print("\n========\n")
                print(joint_col_name)
                sorted_dict = sorted(mi_scores[joint_col_name].items(), key=lambda item: -item[1])[:50]
                for k, v in sorted_dict:
                    c = column_names[k]
                    print("Feature:", c, "Val:", v)
                    print("C1:", np.round(np.mean(Xi[:,k]),3), "C2:", np.round(np.mean(Xj[:,k]), 3))

    print(mi_scores)


def HTFD_Single_Cluster_Analysis(X_train, y_train, cluster_ids, column_names):
    print("\nCluster Wise discriminative features (HTFD)")
    cluster_entrpy = 0
    cntr = 0
    n_columns = X_train.shape[1]
    n_clusters = len(torch.unique(cluster_ids))
    input_dim = X_train.shape[1]
    HTFD_scores = {}
    top_quartile = np.int(n_columns/4)
    for i in range(n_clusters):
        HTFD_scores[i] = {}
        ci = torch.where(cluster_ids == i)[0]
        for c in range(n_columns):
            Xi_c = X_train[ci][:,c]
            Zc = []
            # Collect values from other clusters
            for j in range(n_clusters):
                if i != j:
                    cj = torch.where(cluster_ids == j)[0]
                    Xj_c = X_train[cj][:,c]
                    Zc = np.concatenate([Zc, Xj_c])

            col_entrpy = 0
            p_vals = np.nan_to_num(ttest_ind(Xi_c, Zc, axis=0, equal_var=True))[1]
            HTFD_scores[i][c] = np.round(-np.log(p_vals + np.finfo(float).eps)*0.05, 3)

        print("\n========\n")
        print("|C{}| = {}".format(i, len(ci)))
        print("|C{}| = {}".format(i, np.bincount(y_train[ci])/len(ci)))
        sorted_dict = sorted(HTFD_scores[i].items(), key=lambda item: item[1])[::-1]
        for feature, pval in sorted_dict:
            f = column_names[feature]
            print("Feature:", f, "HTFD score:", pval)
            for cluster_id in range(n_clusters):
                    c_cluster_id = torch.where(cluster_ids == cluster_id)[0]
                    X_cluster_f = X_train[c_cluster_id][:,feature]
                    print("Cluster:", cluster_id, np.round(np.mean(X_cluster_f),3))


def WDFD_Single_Cluster_Analysis(X_train, y_train, cluster_ids, column_names):
    print("\nCluster Wise discriminative features (HTFD)")
    cluster_entrpy = 0
    cntr = 0
    n_columns = X_train.shape[1]
    n_clusters = len(torch.unique(cluster_ids))
    input_dim = X_train.shape[1]
    mi_scores = {}
    for i in range(n_clusters):
        mi_scores[i] = {}
        ci = torch.where(cluster_ids == i)[0]
        for c in range(n_columns):
            Xi_c = X_train[ci][:,c]
            Zc = []
            # Collect values from other clusters
            for j in range(n_clusters):
                if i != j:
                    cj = torch.where(cluster_ids == j)[0]
                    if len(X_train[cj].shape) == 1:
                        Xj_c = X_train[cj].reshape(1,n_features)[:,c]
                    else:
                        Xj_c = X_train[cj][:,c]
                    Zc = np.concatenate([Zc, Xj_c])

            col_entrpy = 0
            # p_vals = np.nan_to_num(ttest_ind(Xi_c, Zc, axis=0, equal_var=True))[1]
            p_vals = -np.nan_to_num(wd(Xi_c, Zc))
            # p_vals = np.nan_to_num(calc_MI(Xi_c, Zc,0))
            mi_scores[i][c] = p_vals

        print("\n========\n")
        print("|C{}| = {}".format(i, len(ci)))
        print("|C{}| = {:.3f}".format(i, np.bincount(y_train[ci])/len(ci)))

        sorted_dict = sorted(mi_scores[i].items(), key=lambda item: item[1])
        for feature, pval in sorted_dict:
            f = column_names[feature]
            print(f, "\t", -pval, end='\t')
            for cluster_id in range(n_clusters):
                    c_cluster_id = torch.where(cluster_ids == cluster_id)[0]
                    X_cluster_f = X_train[c_cluster_id][:,feature]
                    # print(np.round(np.mean(X_cluster_f),3), end='\t')
            print('')


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


class parameters(object):
    def __init__(self, parser):
        self.input_dim = -1
        self.dataset = parser.dataset
        
        # Training parameters
        self.lr = parser.lr
        self.alpha = float(parser.alpha)
        self.wd = parser.wd
        self.batch_size = parser.batch_size
        self.n_epochs = parser.n_epochs
        self.n_runs = parser.n_runs
        self.pre_epoch = parser.pre_epoch
        self.pretrain = parser.pretrain
        self.load_ae = parser.load_ae
        self.classifier = parser.classifier
        self.tol = parser.tol
        self.attention = parser.attention == "True"
        self.ablation = parser.ablation
        self.cluster_balance = parser.cluster_balance

        # Model parameters
        self.lamda = parser.lamda
        self.beta = parser.beta
        self.gamma = parser.gamma
        self.delta = parser.delta
        self.eta = parser.eta
        self.hidden_dims = parser.hidden_dims
        self.latent_dim = self.n_z = parser.n_z
        self.n_clusters = parser.n_clusters
        self.clustering = parser.clustering
        self.n_classes = parser.n_classes

        # Utility parameters
        self.device = parser.device
        self.verbose = parser.verbose
        self.plot = parser.plot
        self.expt = parser.expt
        self.cluster_analysis = parser.cluster_analysis
        self.log_interval = parser.log_interval
        self.pretrain_path = parser.pretrain_path + "/" + self.dataset + ".pth"


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        # assert torch.min(labels, dim=1)[0] >= 0
        # assert torch.max(labels, dim=1)[0] < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

#######################################################
# Evaluate Critiron
#######################################################


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row, col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row, col)]) * 1.0 / y_pred.size


def multi_class_auc(y_true, y_scores, n_classes):
    # If n_classes = 2, then label_binarize doesn't give a 2d array
    y = label_binarize(y_true, classes=list(range(n_classes+1)))[:,:n_classes]
    scores = []
    for i in range(n_classes):
        scores.append(roc_auc_score(y[:,i], y_scores[:,i]))
    return np.avg(scores)


def multi_class_auprc(y_true, y_scores, n_classes):
    # If n_classes = 2, then label_binarize doesn't give a 2d array
    y = label_binarize(y_true, classes=list(range(n_classes+1)))[:,:n_classes]
    scores = []
    for i in range(n_classes):
        scores.append(average_precision_score(y[:,i], y_scores[:,i]))
    return np.avg(scores)


def plot_data(X, y, cluster_ids, args, e):
    reducer = umap.UMAP(random_state=42)
    X2 = reducer.fit_transform(X.cpu().detach().numpy())
    c_clusters = [color[3+int(cluster_ids[i])] for i in range(len(cluster_ids))]
    c_labels = [color[int(y[i])] for i in range(len(cluster_ids))]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Clusters vs Labels')
    ax1.scatter(X2[:,0], X2[:,1], color=c_clusters)
    ax2.scatter(X2[:,0], X2[:,1], color=c_labels)
    fig.savefig(BASE_DIR + "/figures/" + args.dataset + "_e" + str(e) + ".png")
    # plt.show()


def plot(model, X_train, y_train, args, X_test=None, y_test=None, labels=None, epoch=0):
    # idx = torch.Tensor(np.random.randint(0,len(X_train), int(0.1*len(X_train)))).type(torch.LongTensor).to(device)
    idx = range(int(0.2*len(X_train)))
    qs, latents_X = model(X_train[idx], output="latent")
    q_train = qs[0]
    y_train = y_train[idx]

    if labels is not None:
        cluster_id_train = labels[idx]
    else:
        cluster_id_train = torch.argmax(q_train, axis=1)

    print("Training data")
    plot_data(latents_X, y_train, cluster_id_train, args, epoch)

    if X_test is not None:
        qs, latents_test = model(X_test, output="latent")
        q_test = qs[0]
        cluster_id_test = torch.argmax(q_test, axis=1)

        print("Test data")
        plot_data(latents_test, y_test, cluster_id_test, args, epoch)


def drop_constant_column(df):
    """
    Drops constant value columns of pandas dataframe.
    """
    return df.loc[:, (df != df.iloc[0]).any()]


def get_dataset(DATASET, DATA_DIR):
    scale = None
    if DATASET == "cic" or DATASET == "cic_new":
        Xa = pd.read_csv(DATA_DIR + '/' + DATASET + '/' + "cic_set_a.csv")
        Xb = pd.read_csv(DATA_DIR + '/' + DATASET + '/' + "cic_set_b.csv")
        Xc = pd.read_csv(DATA_DIR + '/' + DATASET + '/' + "cic_set_c.csv")

        ya = Xa['In-hospital_death']
        yb = Xb['In-hospital_death']
        yc = Xc['In-hospital_death']

        Xa = Xa.drop(columns=['recordid', 'Survival', 'In-hospital_death'])
        Xb = Xb.drop(columns=['recordid', 'Survival', 'In-hospital_death'])
        Xc = Xc.drop(columns=['recordid', 'Survival', 'In-hospital_death'])

        cols = Xa.columns

        scale = StandardScaler()
        Xa = scale.fit_transform(Xa)
        Xb = scale.transform(Xb)
        Xc = scale.transform(Xc)

        Xa = pd.DataFrame(Xa, columns=cols)
        Xb = pd.DataFrame(Xb, columns=cols)
        Xc = pd.DataFrame(Xc, columns=cols)

        Xa = Xa.fillna(0)
        Xb = Xb.fillna(0)
        Xc = Xc.fillna(0)

        X_train = pd.concat([Xa, Xb])
        y_train = pd.concat([ya, yb])

        X_test = Xc
        y_test = yc

        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test]).to_numpy()
        columns = cols
        X = X.to_numpy()


    elif DATASET == "infant":
        X = pd.read_csv(DATA_DIR + "/" + DATASET + "/" + "X.csv")
        columns = X.columns
        y = pd.read_csv(DATA_DIR + "/" + DATASET + "/" + "y.csv").to_numpy()
        y1 = []
        
        for i in range(len(y)):
            y1.append(y[i][0])

        y = np.array(y1)
        y = y.astype(int)
        enc = OneHotEncoder(handle_unknown='ignore')
        X = enc.fit_transform(X).toarray()
        X = scale.fit_transform(X)

    else:
        X = pd.read_csv(DATA_DIR + "/" + DATASET + "/" + "X.csv")
        columns = X.columns
        y = pd.read_csv(DATA_DIR + "/" + DATASET + "/" + "y.csv").to_numpy()
        y1 = []
        for i in range(len(y)):
            y1.append(y[i][0])
        y = np.array(y1)
        scale = StandardScaler()
        X = scale.fit_transform(X)

    # X, columns = drop_constant_column(X, columns)
    return X, y, columns, scale


def create_imbalanced_data_clusters(n_samples=1000, n_features=8, n_informative=5, n_classes=2,\
                            n_clusters = 2, frac=0.4, outer_class_sep=0.5, inner_class_sep=0.2, clus_per_class=2, seed=0):
    np.random.seed(seed)
    X = np.empty(shape=n_features)
    Y = np.empty(shape=1)
    offsets = np.random.normal(0, outer_class_sep, size=(n_clusters, n_features))
    for i in range(n_clusters):
        samples = int(np.random.normal(n_samples, n_samples/10))
        x, y = make_classification(n_samples=samples, n_features=n_features, n_informative=n_informative,\
                                    n_classes=n_classes, class_sep=inner_class_sep, n_clusters_per_class=clus_per_class)
                                    # n_repeated=0, n_redundant=0)
        x += offsets[i]
        y_0 = np.where(y == 0)[0]
        y_1 = np.where(y != 0)[0]
        y_1 = np.random.choice(y_1, int(np.random.normal(frac, frac/4)*len(y_1)))
        index = np.hstack([y_0,y_1])
        np.random.shuffle(index)
        x_new = x[index]
        y_new = y[index]

        X = np.vstack((X,x_new))
        Y = np.hstack((Y,y_new))

    X = pd.DataFrame(X[1:,:])
    Y = Y[1:]
    columns = ["feature_"+str(i) for i in range(n_features)]
    return X, np.array(Y).astype('int'), columns


def extract_column(X, col_idx, n_classes):
    los_quantiles = np.quantile(X[:,col_idx], np.arange(n_classes+1)/n_classes)
    y_new = []
    for i in range(len(X)):
        lbl = int(X[i,col_idx]/n_classes)
        for j in range(n_classes):
            if los_quantiles[j] <= X[i,col_idx] < los_quantiles[j+1]:
                lbl = j
        y_new.append(lbl)

    X_new = np.delete(X, col_idx, 1) # delete col_idx column
    y_new = np.array(y_new)

    return X_new, y_new


def generate_data_loaders(X, y, batch_size):
    X_data_loader = list(zip(X.astype(np.float32), y, range(len(X))))
    data_loader = torch.utils.data.DataLoader(X_data_loader,\
        batch_size=batch_size, shuffle=True)
    return data_loader


def get_train_val_test_loaders(args, r_state=0):
    if args.dataset in DATASETS:
        if args.dataset != "aki" and args.dataset != "ards" and args.dataset != "cic_los" and args.dataset != "cic_los_new":
            if args.dataset == "synthetic":
                n_feat = 45
                X, y, columns = create_imbalanced_data_clusters(n_samples=5000,\
                       n_clusters=args.n_clusters, n_features = n_feat,\
                       inner_class_sep=0.2, outer_class_sep=5, seed=0)
                args.input_dim = n_feat

            elif args.dataset == "paper_synthetic":
                n_feat = 100
                X, y = paper_synthetic(2500, centers=args.n_clusters)
                args.input_dim = n_feat
                print(args.input_dim)
                scale = None
                columns = ["feature_"+str(i) for i in range(n_feat)]

            else:
                X, y, columns, scale = get_dataset(args.dataset, DATA_DIR)
                args.input_dim = X.shape[1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r_state)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=r_state)

            scale = StandardScaler()
            X_train = scale.fit_transform(X_train)
            X_val = scale.fit_transform(X_val)
            X_test = scale.fit_transform(X_test)

        elif args.dataset == "cic_los" or args.dataset == "cic_los_new":
            if args.dataset == "cic_los":
                X, y, columns, scale = get_dataset("cic", DATA_DIR)
            else:
                X, y, columns, scale = get_dataset("cic_new", DATA_DIR)

            los_quantiles = np.quantile(X[:,2], [0, 0.33, 0.66, 1])
            columns = columns.delete(2)
            y_los = []
            for i in range(len(X)):
                lbl = 0
                if X[i,2] < los_quantiles[1]:
                    lbl = 0
                elif los_quantiles[1] < X[i,2] < los_quantiles[2]:
                    lbl = 1
                elif los_quantiles[2] < X[i,2] < los_quantiles[3]:
                    lbl = 2
                y_los.append(lbl)

            X_los = np.delete(scale.inverse_transform(X), 2, 1) # 2nd column is LOS
            y = np.array(y_los)
            args.input_dim = X_los.shape[1]

            X_train, X_test, y_train, y_test = train_test_split(X_los, y, random_state=r_state)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=r_state)

            scale = StandardScaler()
            X_train = scale.fit_transform(X_train)
            X_val = scale.fit_transform(X_val)
            X_test = scale.fit_transform(X_test)

        elif args.dataset == "aki":
            X, y, columns, scale = get_dataset(args.dataset, DATA_DIR)

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r_state)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=r_state)

            args.input_dim = X_train.shape[1]

        else:
            X, y, columns, scale = get_dataset(args.dataset, DATA_DIR)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r_state)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=r_state)

            args.input_dim = X_train.shape[1]

        return scale, columns, (X_train, y_train), (X_val, y_val), (X_test, y_test)
    else:
        return None


def paper_synthetic(n_pts=1000, centers=4):
    X, y = make_blobs(n_pts, centers=centers)
    W = np.random.randn(10,2)
    U = np.random.randn(100,10)
    X1 = W.dot(X.T)
    X1 = X1*(X1>0)
    X2 = U.dot(X1)
    X2 = X2*(X2>0)
    y = np.random.randint(0,2,len(y))
    return X2.T, y


def performance_metrics(y_true, y_pred, n_classes=2):
    acc_scores, auroc_scores, auprc_scores, minpse_scores = [], [], [], []
    y = label_binarize(y_true, classes=list(range(n_classes+1)))[:,:n_classes]
    cm = metrics.confusion_matrix(y_true, y_pred.argmax(axis=1))
    cm = cm.astype(np.float32)

    for c in range(n_classes):
        tp = cm[c,c]
        fp = sum(cm[:,c]) - cm[c,c]
        fn = sum(cm[c,:]) - cm[c,c]
        tn = sum(np.delete(sum(cm)-cm[c,:],c))
        
        recall = tp/(tp+fn)
        if (tp+fp == 0):
            precision = 0
        else:
            precision = tp/(tp+fp)
        specificity = tn/(tn+fp)

        auroc_scores.append(roc_auc_score(y[:,c], y_pred[:,c]))
        auprc_scores.append(average_precision_score(y[:,c], y_pred[:,c]))
        minpse_scores.append(min(precision, specificity))
        acc_scores.append(accuracy_score(y[:,c], y_pred.argmax(axis=1)))

    return {"acc": np.avg(acc_scores),
            "auroc": np.avg(auroc_scores),
            "auprc": np.avg(auprc_scores),
            "minpse": np.avg(minpse_scores),
            "f1_score":metrics.f1_score(y_true, y_pred.argmax(axis=1), average="macro")}


'''
for d in datasets:
    out = ""
    for k in K:
        if k == 2:
            out = d
        out += " & " + str(k) + " & "
        itr = 0
        for df_idx in range(len(df_list)):
            itr += 1
            df = df_list[df_idx]
            if len(df[(df.Dataset == d) & (df.k == k)]) > 0:
                out += "{$" + str(df[(df.Dataset == d) & (df.k == k)].AUPRC.values[0])
                out += " \\pm "
                out += str(df[(df.Dataset == d) & (df.k == k)].AUPRC_std.values[0])  + "$}"
                if df_idx < len(df_list) - 1:
                    out += " & "
        out += "\\ \n"
    out += "\\\\ \n"
    if k == 4:
        out += "\\midrule \n"
    print(out)

In [22]: for d in datasets:
    out = ""
    for k in K:
     if k == 2:
         out = d
     out += " & " + str(k) + " & "
     itr = 0
     for df_idx in range(len(df_list)):
         itr += 1
         df = df_list[df_idx]
         if len(df[(df.Dataset == d) & (df.k == k)]) > 0:
             out += "{$" + str(df[(df.Dataset == d) & (df.k == k)].AUPRC.values[0])
             out += " \\pm "
             out += str(df[(df.Dataset == d) & (df.k == k)].AUPRC_STD.values[0])  + "$}"
             if df_idx < len(df_list) - 1:
                 out += " & "
     out += "\\\\ \n"
    #out += "\\\\ \n"
    if k == 4:
     out += "\\midrule \n"
    print(out)
    
'''

## Ablation Parameter Ranges ##
alphas = [0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
betas = [0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
gammas = [0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
deltas = [0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
ks = [1, 2, 3, 4, 5]