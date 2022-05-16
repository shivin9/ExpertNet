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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.nn import Linear
from pytorchtools import EarlyStoppingDMNN
from scipy.cluster.vq import kmeans2

import numbers
from sklearn.metrics import davies_bouldin_score as dbs, adjusted_rand_score as ari
from matplotlib import pyplot as plt
color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

from models import DMNN
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
parser.add_argument('--n_z', default= 32, type=int)
parser.add_argument('--n_clusters', default= 3, type=int)
parser.add_argument('--clustering', default= 'cac')
parser.add_argument('--n_classes', default= 2, type=int)

# Utility parameters
parser.add_argument('--device', default= 'cpu')
parser.add_argument('--log_interval', default= 10, type=int)
parser.add_argument('--verbose', default= 'False')
parser.add_argument('--plot', default= 'False')
parser.add_argument('--other', default= 'False')
parser.add_argument('--cluster_analysis', default= 'False')
parser.add_argument('--pretrain_path', default= '/Users/shivin/Document/NUS/Research/CAC/CAC_DL/ExpertNet/pretrained_model')


parser = parser.parse_args()
args = parameters(parser)

####################################################################################
####################################################################################
####################################################################################
###################################### Training ####################################
####################################################################################
####################################################################################
####################################################################################

criterion = nn.CrossEntropyLoss(reduction='mean')

f1_scores, auc_scores, auprc_scores, acc_scores = [], [], [], []
sil_score, HTFD_score, wdfd_score = 0, 0, 0

if args.verbose == "False":
    blockPrint()

for r in range(args.n_runs):
    scale, column_names, train_data, val_data, test_data = get_train_val_test_loaders(args, r_state=r)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    train_loader = generate_data_loaders(X_train, y_train, args.batch_size)
    val_loader = generate_data_loaders(X_val, y_val, args.batch_size)
    test_loader = generate_data_loaders(X_test, y_test, args.batch_size)

    # Architecture used in ExpertNet paper
    # ae_layers = [64, 32, args.n_z, 32, 64]
    # expert_layers = [args.n_z, 64, 32, 16, 8, args.n_classes]

    # Architecture for CAC paper
    ae_layers = [64, args.n_z, 64]
    expert_layers = [args.n_z, 30, args.n_classes]

    model = DMNN(ae_layers, expert_layers, args=args).to(args.device)
    device = args.device

    N_EPOCHS = args.n_epochs
    es = EarlyStoppingDMNN(dataset=args.dataset)
    model.pretrain(train_loader, args.pretrain_path)
    optimizer = Adam(model.parameters(), lr=args.lr)

    z_train, _, gate_vals = model(torch.FloatTensor(X_train))
    original_cluster_centers, cluster_indices = kmeans2(z_train.data.cpu().numpy(), k=args.n_clusters, minit='++')
    one_hot_cluster_indices = label_binarize(cluster_indices, classes=list(range(args.n_clusters+1)))[:,:args.n_clusters]
    one_hot_cluster_indices = torch.FloatTensor(one_hot_cluster_indices)

    # train gating network
    for e in range(1, N_EPOCHS):
        z_batch, _, gate_vals = model(torch.FloatTensor(X_train))
        model.train()
        optimizer.zero_grad()
        gating_err = criterion(gate_vals, one_hot_cluster_indices)
        gating_err.backward()
        optimizer.step()

    cluster_ids_train = torch.argmax(gate_vals, axis=1)
    sil_score  = silhouette_new(z_train.detach().numpy(), cluster_ids_train, metric='euclidean')
    HTFD_score = calculate_HTFD(torch.FloatTensor(X_train), cluster_ids_train)
    wdfd_score = calculate_WDFD(torch.FloatTensor(X_train), cluster_ids_train)

    # train Local Experts
    for e in range(1, N_EPOCHS):
        epoch_loss = 0
        epoch_auc = 0
        epoch_f1 = 0
        auc = 0
        model.train()
        for X_batch, y_batch, idx in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            z_batch, x_bar, gate_vals = model(X_batch)
            train_loss = torch.tensor(0.).to(args.device)
            y_pred = torch.zeros((len(X_batch), args.n_classes))

            for j in range(args.n_clusters):
                # cluster_ids = np.where(cluster_indices[idx] == j)[0]
                cluster_ids = range(len(X_batch))
                z_cluster, y_cluster = z_train[cluster_ids], torch.Tensor(y_train[cluster_ids]).type(torch.LongTensor)
                preds_j = model.classifiers[j][0](z_cluster.detach())
                optimizer_j = model.classifiers[j][1]
                optimizer_j.zero_grad()
                loss_j = nn.CrossEntropyLoss(reduction='mean')(preds_j, y_cluster)
                local_loss = torch.sum(gate_vals.detach()[cluster_ids,j]*loss_j)
                local_loss.backward(retain_graph=True)
                optimizer_j.step()
                y_pred[cluster_ids] += torch.reshape(gate_vals[cluster_ids,j], shape=(len(preds_j), 1)) * preds_j
                train_loss += local_loss

            # print(train_loss)
            epoch_loss += train_loss
            # train_loss.backward(retain_graph=True)
            f1 = f1_score(np.argmax(y_pred.detach().numpy(), axis=1), y_batch.detach().numpy(), average="macro")
            auc = multi_class_auc(y_batch.detach().numpy(), y_pred.detach().numpy(), args.n_classes)
            auprc = multi_class_auprc(y_batch.detach().numpy(), y_pred.detach().numpy(), args.n_classes)
            epoch_auc += auc.item()
            epoch_f1 += f1.item()

        z_val, x_bar, gate_vals = model(torch.FloatTensor(X_val).to(args.device))
        val_pred = torch.zeros((len(X_val), args.n_classes))
        for j in range(args.n_clusters):
            model.classifiers[j][0].eval()
            preds_j = model.classifiers[j][0](z_val)
            loss_j = nn.CrossEntropyLoss(reduction='mean')(preds_j, torch.Tensor(y_val).type(torch.LongTensor))
            train_loss += torch.sum(gate_vals[:,j]*loss_j)
            val_pred += torch.reshape(gate_vals[:,j], shape=(len(preds_j), 1)) * preds_j

        val_loss = nn.CrossEntropyLoss(reduction='mean')(val_pred, torch.tensor(y_val).to(device))
        val_f1 = f1_score(np.argmax(val_pred.detach().numpy(), axis=1), y_val, average="macro")
        val_auc = multi_class_auc(y_val, val_pred.detach().numpy(), args.n_classes)
        val_auprc = multi_class_auprc(y_val, val_pred.detach().numpy(), args.n_classes)
        es([val_f1, val_auprc], model)

        print(f'Epoch {e+0:03}: | Train Loss: {epoch_loss/len(train_loader):.5f} | ',
        	f'Train F1: {epoch_f1/len(train_loader):.3f} | Train AUC: {epoch_auc/len(train_loader):.3f} | ',
        	f'Val F1: {val_f1:.3f} | Val Auc: {val_auc:.3f} | Val Loss: {val_loss:.3f}')

        if es.early_stop == True:
            break


    ####################################################################################
    ####################################################################################
    ####################################################################################
    ###################################### Testing #####################################
    ####################################################################################
    ####################################################################################
    ####################################################################################

    print("\n####################################################################################\n")
    print("Evaluating Test Data")

    # Load best model trained from local training phase
    model = es.load_checkpoint(model)
    z_test, _, gate_test = model(torch.FloatTensor(np.array(X_test)).to(args.device))
    test_pred = torch.zeros((len(X_test), args.n_classes))

    for j in range(args.n_clusters):
        model.classifiers[j][0].eval()
        preds_j = model.classifiers[j][0](z_test)
        loss_j = nn.CrossEntropyLoss(reduction='mean')(preds_j, torch.Tensor(y_test).type(torch.LongTensor))
        train_loss += torch.sum(gate_test[:,j]*loss_j)
        test_pred += torch.reshape(gate_test[:,j], shape=(len(preds_j), 1)) * preds_j

    test_loss = nn.CrossEntropyLoss(reduction='mean')(test_pred, torch.tensor(y_test).to(device))
    test_f1 = f1_score(np.argmax(test_pred.detach().numpy(), axis=1), y_test, average="macro")
    test_auc = multi_class_auc(y_test, test_pred.detach().numpy(), args.n_classes)
    test_auprc = multi_class_auprc(y_test, test_pred.detach().numpy(), args.n_classes)
    test_acc = accuracy_score(np.argmax(test_pred.detach().numpy(), axis=1), y_test)
    es([val_f1, val_auprc], model)

    y_preds = np.argmax(test_pred.detach().numpy(), axis=1)
    print(confusion_matrix(y_test, y_preds))

    print(f'Epoch {e+0:03}: | Train Loss: {epoch_loss/len(train_loader):.5f} | ',
    	f'Train F1: {epoch_f1/len(train_loader):.3f} | Train AUC: {epoch_auc/len(train_loader):.3f}| ',
    	f'Test F1: {test_f1:.3f} | Test AUC: {test_auc:.3f} | Test Loss: {test_loss:.3f}')

    print("\n####################################################################################\n")
    f1_scores.append(test_f1)
    auc_scores.append(test_auc)
    auprc_scores.append(test_auprc)
    acc_scores.append(test_acc)


enablePrint()
print("F1:", f1_scores)
print("AUC:", auc_scores)
print("AUPRC:", auprc_scores)
print("ACC:", acc_scores)

print("[Avg]\tDataset\tk\tF1\tAUC\tAUPRC\tACC\tSIL\tHTFD\tWDFD")

print("\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format\
    (args.dataset, args.n_clusters, np.avg(f1_scores), np.avg(auc_scores),\
    np.avg(auprc_scores), np.avg(acc_scores), sil_score, HTFD_score, wdfd_score))

print("[Std]\tF1\tAUC\tAUPRC\tACC\tSIL\tHTFD\tWDFD")

print("\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format\
    (np.std(f1_scores), np.std(auc_scores),\
    np.std(auprc_scores), np.std(acc_scores), 0, 0, 0))
