import pandas as pd
import numpy as np
import argparse
import os
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import itertools
import random
import warnings
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, precision_recall_fscore_support,\
roc_auc_score, accuracy_score, f1_score
from sklearn.utils import class_weight, shuffle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import tensorflow as tf
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras import models, layers, losses, optimizers, initializers, regularizers
from keras.utils.vis_utils import plot_model
from keras import backend
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default= 'creditcard')
parser.add_argument('--input_dim', default= '-1')

# Training parameters
parser.add_argument('--alpha', default= 1, type=float)
parser.add_argument('--batch_size', default= 512, type=int)
parser.add_argument('--cv', default= "False")
parser.add_argument('--n_epochs', default= 100)

# Model parameters
parser.add_argument('--hidden_dims', default= [64, 32])
parser.add_argument('--n_z', default= 20, type=int)
parser.add_argument('--n_clusters', default= 3, type=int)
parser.add_argument('--clustering', default= 'cac')
parser.add_argument('--n_classes', default= 2, type=int)

# Utility parameters
parser.add_argument('--device', default= 'cpu')
parser.add_argument('--log_interval', default= 10, type=int)
parser.add_argument('--pretrain_path', default= '/Users/shivin/Document/NUS/Research/CAC/CAC_DL/DeepCAC/pretrained_model')

args = parser.parse_args()

test_results = pd.DataFrame(columns=['Dataset', 'Classifier', 'alpha', 'k', \
    'Base_F1_mean', 'Base_AUC_mean','Base_F1_std', 'Base_AUC_std',\
    'KM_F1_mean', 'KM_AUC_mean', 'KM_F1_std', 'KM_AUC_std',\
    'CAC_F1_mean', 'CAC_AUC_mean', 'CAC_F1_std', 'CAC_AUC_std'], dtype=object)

res = pd.DataFrame(columns=['Dataset', 'Classifier', 'alpha', 'k', \
    'Base_F1_mean', 'Base_AUC_mean',\
    'KM_F1_mean', 'KM_AUC_mean', \
    'CAC_F1_mean', 'CAC_AUC_mean'], dtype=object)

# DATASET = "sepsis" # see folder, *the Titanic dataset is different*
DATASET = args.dataset
alphas = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1]

scale, column_names, train_data, val_data, test_data = get_train_val_test_loaders(args)
X_train, y_train, train_loader = train_data
X_val, y_val, val_loader = val_data
X_test, y_test, test_loader = test_data


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def neural_network(X_train, y_train, X_val, y_val, X_test, y_test, n_experts, cluster_algo, args, n_classes=2):
    data_len = X_train.shape[1]
    ## Define Neural Network
    experts = []
    inputTensor = layers.Input(shape=(data_len,))
    # encode = layers.Dense(units=64, name='encode_1', activation=None, activity_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(inputTensor)
    # encode = layers.Dense(units=32, name='encode_2', activation=None, activity_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(encode)
    # decode = layers.Dense(units=64, name='decode_2', activation=None)(encode)
    # decode = layers.Dense(units=data_len, name='decode_1', activation=None)(decode)

    encode = layers.Dense(units=128, name='encode_0', activation=None)(inputTensor)
    encode = layers.Dense(units=64, name='encode_1', activation=None)(encode)
    encode = layers.Dense(units=32, name='encode_2', activation=None)(encode)    
    embed  = layers.Dense(units=args.n_z, name='embed', activation=None)(encode)
    decode = layers.Dense(units=32, name='decode_3', activation=None)(embed)
    decode = layers.Dense(units=64, name='decode_2', activation=None)(decode)
    decode = layers.Dense(units=128, name='decode_1', activation=None)(decode)
    decode = layers.Dense(units=data_len, name='decode_0', activation=None)(decode)
    gate   = layers.Dense(n_experts*n_classes, activation='softmax', name='gating')(embed)

    for i in range(n_experts):
      layer_var = layers.Dense(16, activation='relu', name='dense_{}_2'.format(i))(embed)
      layer_var = layers.Dense(8, activation='relu', name='dense_{}_3'.format(i))(layer_var)
      layer_var = layers.Dense(n_classes, activation='relu', name='dense_{}_4'.format(i))(layer_var)
      experts.append(layer_var)
      del layer_var

    if n_experts == 1:
      outputTensor = experts

    else: 
      mergedTensor = layers.Concatenate(axis=1)(experts)
      outputTensor = layers.Dot(axes=1)([gate, mergedTensor])

    # Define autoencoder
    dae = models.Model(inputs=inputTensor, outputs=decode)
    dae.compile(
        optimizer=optimizers.Adadelta(learning_rate=0.1),
        loss='MeanSquaredError'
    )

    # Define cluster gating
    cluster = models.Model(inputs=inputTensor, outputs=gate)
    for i in ['encode_0', 'encode_1', 'encode_2', 'embed']:
      cluster.get_layer(i).trainable = False

    cluster.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
    )

    # Define full model
    full = models.Model(inputs=inputTensor, outputs=outputTensor)
    for i in ['encode_0', 'encode_1', 'encode_2', 'embed']:
      full.get_layer(i).trainable = True

    full.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['categorical_crossentropy'],
    )

    ## Train autoencoder
    history_dae = dae.fit(
        x=X_train+np.random.normal(0,0.05,X_train.shape),
        y=X_train,
        batch_size=args.batch_size,
        shuffle=True,
        epochs=100,
        use_multiprocessing=True,
        verbose=0
        )

    test_loss = dae.evaluate(x=X_test, y=X_test)
    # print('Autoencoder loss:', test_loss)

    ## Get embeddings
    encoder = models.Model(inputs=inputTensor, outputs=encode)
    X_train_embeddings = encoder.predict(x=X_train)

    if cluster_algo == 'KMeans':
      ## KMeans Clustering
      cluster_alg = KMeans(n_clusters=n_experts, random_state=0)
      X_train_clusters = cluster_alg.fit_predict(X_train_embeddings)
      temp = np.zeros(n_classes*len(X_train_clusters))
      for i in range(len(temp)):
        temp[i] = X_train_clusters[int(i/2)]
    else:
      raise ValueError('Method not supported')


    X_train_clusters_sparse = MultiLabelBinarizer().fit_transform(X_train_clusters.reshape(-1,1))

    ## Train cluster gating
    history_cluster = cluster.fit(
        x=X_train,
        y=X_train_clusters_sparse,
        batch_size=1000,
        shuffle=True,
        epochs=100,
        use_multiprocessing=True,
        verbose=0
        )

    ## Check cluster gating accuracy
    cluster_ids_train = np.argmax(cluster.predict(X_train), axis=1)
    # scores = precision_recall_fscore_support(X_train_clusters_sparse, y_pred > 0.5, average='weighted')
    sil_score = silhouette_new(X_train_embeddings, cluster_ids_train, metric='euclidean')
    nhfd_score = calculate_nhfd(X_train, torch.tensor(cluster_ids_train))
    wdfd_score = calculate_WDFD(X_train, torch.tensor(cluster_ids_train))

    ## Train full model
    history_full = full.fit(
        x=X_train,
        y=y_train,
        batch_size=args.batch_size,
        shuffle=True,
        epochs=args.n_epochs,
        use_multiprocessing=True,
        verbose=0
        )

    ## Check full model accuracy
    y_pred = full.predict(X_test)

    # scores = precision_recall_fscore_support(y_test, y_pred > 0.5, average='weighted')
    # precision_ls = scores[0]
    # recall_ls = scores[1]
    fscore_ls = f1_score(y_test, y_pred > 0.5)
    accuracy_ls = accuracy_score(y_test, y_pred > 0.5)

    if len(np.unique(y_test)) == 1:
      auroc_ls = 0
    else:
      auroc_ls = roc_auc_score(y_test, y_pred)

    backend.clear_session()

    return {'f1_score': fscore_ls,
          'auroc': auroc_ls,
          'accuracy': accuracy_ls,
          'sil_score': sil_score,
          'nhfd_score': nhfd_score,
          'wdfd_score': wdfd_score}


param_grid = {
'alpha': alphas,
# 'k': [2, 3, 4, 5]
'k': [2]
}


params = {'titanic': [0.2, 2],
  'magic': [0.01, 2],
  'creditcard': [0.3, 3],
  'adult': [0.8, 2],
  'diabetes': [0.15, 2],
  'cic': [0.1, 2],
  'wid_mortality': [0.1, 2]
}


best_alpha = 0
best_score = 0
test_f1_auc = [0, 0, 0, 0, 0]
keys, values = zip(*param_grid.items())
combs = list(itertools.product(*values))
# random.shuffle(combs)
n_splits = 5
scale = StandardScaler()
res_idx = 0

if args.cv == "False":
    n_clusters = args.n_clusters

    print("Testing on ", DATASET, " with n_clusters = ", args.n_clusters)

    n_runs = 5
    km_scores = np.zeros((n_runs, 5))

    for i in range(n_runs):
        scores_km = neural_network(X_train, y_train, X_val, y_val, X_test, y_test, args.n_clusters, 'KMeans', args)

        km_scores[i, 0] = scores_km['f1_score']
        km_scores[i, 1] = scores_km['auroc']
        km_scores[i, 2] = scores_km['sil_score']
        km_scores[i, 3] = scores_km['nhfd_score']
        km_scores[i, 4] = scores_km['wdfd_score']

    print("k\tF1\tAUC\tSIL\tNHFD\tWDFD")
    print(n_clusters, np.mean(km_scores, axis=0))

else:
    for v in combs:
        hyperparameters = dict(zip(keys, v)) 
        alpha = hyperparameters['alpha']
        n_clusters = params[DATASET][1] # Fix number of clusters to previous values
        print(hyperparameters)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=108)

        base_scores = np.zeros((n_splits, 2))
        km_scores = np.zeros((n_splits, 2))
        cac_scores = np.zeros((n_splits, 2))
        i = 0

        X1, X_test, y1, y_test = train_test_split(X, y, stratify=y, random_state=108)

        # Grid CV happening here
        print("Starting GridCV search")
        for train, val in skf.split(X1, y1):
            print("Iteration: ", i)
            X_train, X_val, y_train, y_val = X1[train], X1[val], y1[train], y1[val]
            X_train = scale.fit_transform(X_train)
            X_val = scale.fit_transform(X_val)

            scores_base = neural_network(X_train, y_train, X_test, y_test, 1, 'KMeans', alpha)
            # scores_cac = neural_network(X_train, y_train, X_test, y_test, n_clusters, 'CAC', alpha)
            scores_km = neural_network(X_train, y_train, X_test, y_test, n_clusters, 'KMeans', alpha)

            base_scores[i, 0] = scores_base['f1_score']
            base_scores[i, 1] = scores_base['auroc']

            km_scores[i, 0] = scores_km['f1_score']
            km_scores[i, 1] = scores_km['auroc']

            # cac_scores[i, 0] = scores_cac['f1_score']
            # cac_scores[i, 1] = scores_cac['auroc']
            i += 1

        print("5-Fold Base scores", np.mean(base_scores, axis=0))
        print("5-Fold KMeans scores", np.mean(km_scores, axis=0))        
        print("5-Fold terminal CAC scores", np.mean(cac_scores, axis=0))
        print("\n")

        res.loc[res_idx] = [DATASET, "DMNN", alpha, n_clusters] + list(np.mean(base_scores, axis=0)) + \
        list(np.mean(km_scores, axis=0)) + \
        list(np.mean(cac_scores, axis=0))
        res_idx += 1
        res.to_csv("./Results/Tuning_every_run" + args.dataset + ".csv", index=None)

        X1 = scale.fit_transform(X1)
        X_test = scale.fit_transform(X_test)

        print("Testing on Test data with alpha = ", alpha)

        # scores_cac = neural_network(X1, y1, X_test, y_test, n_clusters, 'CAC', alpha)
        scores_base = neural_network(X1, y1, X_test, y_test, 1, 'KMeans', alpha)
        scores_km = neural_network(X1, y1, X_test, y_test, n_clusters, 'KMeans', alpha)

        print("Base final test performance: ", "F1: ", scores_base['f1_score'], "AUC: ", scores_base['auroc'], alpha)
        print("KM final test performance: ", "F1: ", scores_km['f1_score'], "AUC: ", scores_km['auroc'], alpha)
        # print("CAC final test performance: ", "F1: ", scores_cac['f1_score'], "AUC: ", scores_cac['auroc'], alpha)
        print("\n")

        # Can choose whether it to do it w.r.t F1 or AUC
        if np.mean(cac_scores, axis=0)[0] > best_score:
            best_score = np.mean(cac_scores, axis=0)[0]
            best_alpha = alpha
            best_k = n_clusters
            test_f1_auc = [scores_base['f1_score'], scores_base['auroc'], scores_km['f1_score'], scores_km['auroc'], scores_cac['f1_score'], scores_cac['auroc']]
            # print(test_f1_auc)
        
    print(DATASET, ": Best alpha = ", best_alpha)
    test_results.loc[0] = [DATASET, "DMNN", best_alpha, best_k] + test_f1_auc
    print(test_results)
    test_results.to_csv("./Results/Tuned_DMNN_Test_Results_" + args.dataset + "" + ".csv", index=None)

    # result_1 = neural_network(1, 'KMeans')
    # result_2 = neural_network(2, 'KMeans')
    # result_3 = neural_network(2, 'CAC')

    # print(result_1)
    # print(result_2)
    # print(result_3)

