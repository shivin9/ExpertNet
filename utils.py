## Utils.py
from __future__ import division, print_function
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, label_binarize
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.stats import ttest_ind, wasserstein_distance as wd
from read_patients import get_aki
from matplotlib import pyplot as plt
import sys
import umap

color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']
DATASETS = ['diabetes', 'ards', 'cic', 'sepsis', 'aki', 'infant', 'wid_mortality', 'synthetic', 'cic_los',\
            'titanic', 'magic', 'adult', 'creditcard']

DATA_DIR = "/Users/shivin/Document/NUS/Research/Data"
BASE_DIR = "/Users/shivin/Document/NUS/Research/cac/cac_dl/DeepCAC"

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
        if len(ci) == 0:
            return 0
        # Collect features of all the columns
        for c in range(n_features):
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


def HTFD_Cluster_Analysis(X_train, cluster_ids, column_names):
    print("\nCluster Wise discriminative features (HTFD)")
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
                p_vals = np.nan_to_num(ttest_ind(Xi, Xj, axis=0, equal_var=True))[1]
                for c in range(n_columns):
                    mi_scores[joint_col_name][c] = np.round(p_vals[c], 3)
                    # print(column_names[c], ":", c_entropy)
                cntr += 1
                print("\n========\n")
                print(joint_col_name)
                sorted_dict = sorted(mi_scores[joint_col_name].items(), key=lambda item: -item[1])[:10]
                for k, v in sorted_dict:
                    c = column_names[k]
                    print("Feature:", c, "Val:", v)
                    print("C1:", np.round(np.mean(Xi[:,k]),3), "C2:", np.round(np.mean(Xj[:,k]), 3))


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
            HTFD_scores[i][c] = np.round(np.exp(-p_vals/0.05), 3)

        print("\n========\n")
        print("|C{}| = {}".format(i, len(ci)))
        print("|C{}| = {}".format(i, np.bincount(y_train[ci])/len(ci)))
        sorted_dict = sorted(HTFD_scores[i].items(), key=lambda item: item[1])[::-1]
        for feature, pval in sorted_dict:
            f = column_names[feature]
            print("Feature:", f, "PVal:", pval)
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
        self.cluster_analysis = parser.cluster_analysis
        self.log_interval = parser.log_interval
        self.pretrain_path = parser.pretrain_path + "/" + self.dataset + ".pth"


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
    return np.average(scores)


def plot(model, X_train, y_train, X_test=None, y_test=None, labels=None):
    reducer = umap.UMAP(random_state=42)
    # idx = torch.Tensor(np.random.randint(0,len(X_train), int(0.1*len(X_train)))).type(torch.LongTensor).to(device)
    idx = range(int(0.2*len(X_train)))
    qs, latents_X = model(X_train[idx], output="latent")
    q_train = qs[0]
    y_train = y_train[idx]

    if labels is not None:
        cluster_id_train = labels[idx]
    else:
        cluster_id_train = torch.argmax(q_train, axis=1)

    X2 = reducer.fit_transform(latents_X.cpu().detach().numpy())

    print("Training data")

    c_clusters = [color[int(cluster_id_train[i])] for i in range(len(cluster_id_train))]
    c_labels = [color[int(y_train[i])] for i in range(len(cluster_id_train))]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Clusters vs Labels')
    ax1.scatter(X2[:,0], X2[:,1], color=c_clusters)
    ax2.scatter(X2[:,0], X2[:,1], color=c_labels)
    plt.show()
    if X_test is not None:
        qs, latents_test = model(X_test, output="latent")
        q_test = qs[0]
        X2 = reducer.transform(latents_test.cpu().detach().numpy())
        cluster_id_test = torch.argmax(q_test, axis=1)
        c_clusters = [color[int(cluster_id_test[i])] for i in range(len(cluster_id_test))]
        c_labels = [color[int(y_test[i])] for i in range(len(cluster_id_test))]

        print("Test data")
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Clusters vs Labels')
        ax1.scatter(X2[:,0], X2[:,1], color=c_clusters)
        ax2.scatter(X2[:,0], X2[:,1], color=c_labels)
        plt.show()


def drop_constant_column(df):
    """
    Drops constant value columns of pandas dataframe.
    """
    return df.loc[:, (df != df.iloc[0]).any()]


def get_dataset(DATASET, DATA_DIR):
    scale = None
    if DATASET == "cic":
        Xa = pd.read_csv(DATA_DIR + "/CIC/cic_set_a.csv")
        Xb = pd.read_csv(DATA_DIR + "/CIC/cic_set_b.csv")
        Xc = pd.read_csv(DATA_DIR + "/CIC/cic_set_c.csv")

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

    else:
        X = pd.read_csv(DATA_DIR + "/" + DATASET + "/" + "X.csv")
        columns = X.columns
        y = pd.read_csv(DATA_DIR + "/" + DATASET + "/" + "y.csv").to_numpy()
        y1 = []
        for i in range(len(y)):
            y1.append(y[i][0])
        y = np.array(y1)

    # X, columns = drop_constant_column(X, columns)
    X = X.to_numpy()
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


def get_train_val_test_loaders(args):
    if args.dataset in DATASETS:
        if args.dataset != "aki" and args.dataset != "ards" and args.dataset != "cic_los":
            if args.dataset == "synthetic":
                n_feat = 45
                X, y, columns = create_imbalanced_data_clusters(n_samples=5000,\
                       n_clusters=args.n_clusters, n_features = n_feat,\
                       inner_class_sep=0.2, outer_class_sep=2, seed=0)
                args.input_dim = n_feat

            elif args.dataset == "paper_synthetic":
                n_feat = 100
                X, y = paper_synthetic(2500, centers=4)
                args.input_dim = n_feat
                print(args.input_dim)

            else:
                X, y, columns, scale = get_dataset(args.dataset, DATA_DIR)
                print(args.dataset)
                args.input_dim = X.shape[1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0)

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_val = sc.fit_transform(X_val)
            X_test = sc.fit_transform(X_test)
            X_train_data_loader = list(zip(X_train.astype(np.float32), y_train, range(len(X_train))))
            X_val_data_loader = list(zip(X_val.astype(np.float32), y_val, range(len(X_val))))
            X_test_data_loader  = list(zip(X_test.astype(np.float32), y_test, range(len(X_train))))

        elif args.dataset == "cic_los":
            X, y, columns, scale = get_dataset("cic", DATA_DIR)
            print(args.dataset)

            los_quantiles = np.quantile(X[:,2], [0, 0.33, 0.66, 1])
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

            X_los = np.delete(X, 2, 1) # 2nd column is LOS
            y = np.array(y_los)
            args.input_dim = X_los.shape[1]

            X_train, X_test, y_train, y_test = train_test_split(X_los, y, random_state=0)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0)

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_val = sc.fit_transform(X_val)
            X_test = sc.fit_transform(X_test)
            X_train_data_loader = list(zip(X_train.astype(np.float32), y_train, range(len(X_train))))
            X_val_data_loader = list(zip(X_val.astype(np.float32), y_val, range(len(X_val))))
            X_test_data_loader  = list(zip(X_test.astype(np.float32), y_test, range(len(X_train))))


        elif args.dataset == "aki":
            print("Loading aki Train")
            X, y, columns, scale = get_dataset(args.dataset, DATA_DIR)

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0)

            args.input_dim = X_train.shape[1]

            X_train_data_loader = list(zip(X_train.astype(np.float32), y_train, range(len(X_train))))
            X_val_data_loader = list(zip(X_val.astype(np.float32), y_val, range(len(X_val))))
            X_test_data_loader  = list(zip(X_test.astype(np.float32), y_test, range(len(X_train))))

        else:
            print("Loading ards Train")
            X, y, columns, scale = get_dataset(args.dataset, DATA_DIR)

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0)

            args.input_dim = X_train.shape[1]

            X_train_data_loader = list(zip(X_train.astype(np.float32), y_train, range(len(X_train))))
            X_val_data_loader = list(zip(X_val.astype(np.float32), y_val, range(len(X_val))))
            X_test_data_loader  = list(zip(X_test.astype(np.float32), y_test, range(len(X_train))))


        train_loader = torch.utils.data.DataLoader(X_train_data_loader,
            batch_size=args.batch_size, shuffle=True)

        val_loader = torch.utils.data.DataLoader(X_val_data_loader,
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(X_test_data_loader, 
            batch_size=args.batch_size, shuffle=False)

        return scale, columns, (X_train, y_train, train_loader), (X_val, y_val, val_loader), (X_test, y_test, test_loader)
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
    return X2.T, y

## Ablation Parameter Ranges ##
alphas = [0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
betas = [0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
gammas = [0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
deltas = [0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
ks = [1, 2, 3, 4, 5]