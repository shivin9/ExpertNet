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

from models import DMNN, CNN_AE, DAE, CIFAR_AE, target_distribution
from utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default= 'creditcard')
parser.add_argument('--input_dim', default= '-1')
parser.add_argument('--n_features', default= '-1')
parser.add_argument('--target', default= -1, type=int)
parser.add_argument('--data_ratio', default= -1, type=float)

# Training parameters
parser.add_argument('--lr_enc', default= 0.002, type=float)
parser.add_argument('--lr_exp', default= 0.002, type=float)
parser.add_argument('--alpha', default= 1, type=float)
parser.add_argument('--wd', default= 5e-4, type=float)
parser.add_argument('--batch_size', default= 512, type=int)
parser.add_argument('--n_epochs', default= 10, type=int)
parser.add_argument('--n_runs', default= 5, type=int)
parser.add_argument('--pre_epoch', default= 75, type=int)
parser.add_argument('--pretrain', default= True, type=bool)
parser.add_argument("--load_ae",  default=False, type=bool)
parser.add_argument("--classifier", default="LR")
parser.add_argument("--tol", default=0.01, type=float)
parser.add_argument("--attention", default="True")
parser.add_argument('--ablation', default='None')
parser.add_argument('--cluster_balance', default='hellinger')
parser.add_argument('--optimize', default= 'auprc')
parser.add_argument('--ae_type', default= 'dae')
parser.add_argument('--sub_epochs', default= 'False')
parser.add_argument('--n_channels', default= 1, type=int)

# Model parameters
parser.add_argument('--lamda', default= 1, type=float)
parser.add_argument('--beta', default= 0.5, type=float) # KM loss wt
parser.add_argument('--gamma', default= 1.0, type=float) # Classification loss wt
parser.add_argument('--delta', default= 0.01, type=float) # Class seploss wt
parser.add_argument('--eta', default= 0.01, type=float) # Cluster Balance loss
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
parser.add_argument('--expt', default= 'ExpertNet') # Running DMNN for which experiment?
parser.add_argument('--other', default= 'False')
parser.add_argument('--cluster_analysis', default= 'False')
parser.add_argument('--pretrain_path', default= '/Users/shivin/Document/NUS/Research/CAC/CAC_DL/ExpertNet/pretrained_model/DMNN')


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

f1_scores, auc_scores, auprc_scores, minpse_scores, acc_scores = [], [], [], [], []
sil_scores, nmi_scores, ari_scores, HTFD_scores, wdfd_scores = [], [], [], [], []
test_cluster_nos = []

if args.verbose == "False":
    blockPrint()

base_suffix = ""
base_suffix += args.dataset
base_suffix += "_" + args.ae_type
base_suffix += "_k_" + str(args.n_clusters)
base_suffix += "_att_" + str(args.attention)
base_suffix += "_dr_" + str(args.data_ratio)
base_suffix += "_target_" + str(args.target)
es_path = args.pretrain_path + base_suffix
args.pretrain_path += "/AE_" + base_suffix + ".pth"

for r in range(args.n_runs):
    scale, column_names, train_data, val_data, test_data = get_train_val_test_loaders(args, r_state=r, n_features=args.n_features)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    train_loader = generate_data_loaders(X_train, y_train, args.batch_size)
    val_loader = generate_data_loaders(X_val, y_val, args.batch_size)
    test_loader = generate_data_loaders(X_test, y_test, args.batch_size)

    if args.expt == 'ExpertNet':
        ae_layers = [128, 64, args.n_z, 64, 128]
        expert_layers = [args.n_z, 128, 64, 32, 16, args.n_classes]

    else:
        # DeepCAC expts
        ae_layers = [64, args.n_z, 64]
        expert_layers = [args.n_z, 30, args.n_classes]

    if args.ae_type == 'cnn':
        if X_train[0].shape[1] == 28:
            dmnn_ae = CNN_AE(args, fc2_input_dim=128)
        elif X_train[0].shape[1] == 32:
            dmnn_ae = CIFAR_AE(args, fc2_input_dim=128)
    
    else:
        ae_layers.append(args.input_dim)
        ae_layers = [args.input_dim] + ae_layers
        dmnn_ae = DAE(ae_layers)

    model = DMNN(dmnn_ae,
            expert_layers,
            args.lr_enc,
            args.lr_exp,
            args=args).to(args.device)
    device = args.device

    N_EPOCHS = args.n_epochs
    es = EarlyStoppingDMNN(dataset=args.dataset, path=es_path)
    model.pretrain(train_loader, args.pretrain_path)
    # params = list(model.ae.parameters()) + list(model.gate.parameters())
    params = list(model.parameters())
    optimizer = Adam(params, lr=args.lr_enc)

    z_train, _, gate_vals = model(torch.FloatTensor(X_train))
    original_cluster_centers, cluster_indices = kmeans2(z_train.data.cpu().numpy(), k=args.n_clusters, minit='++')
    one_hot_cluster_indices = label_binarize(cluster_indices, classes=list(range(args.n_clusters+1)))[:,:args.n_clusters]
    one_hot_cluster_indices = torch.FloatTensor(one_hot_cluster_indices)
    print("Pre Pre-training : ", np.bincount(cluster_indices))

    # train gating network
    for e in range(100):
        z_train, _, gate_vals = model(torch.FloatTensor(X_train))
        model.train()
        optimizer.zero_grad()
        gating_err = criterion(gate_vals, one_hot_cluster_indices)

        P = torch.sum(gate_vals, axis=0)
        P = P/P.sum()
        Q = torch.ones(args.n_clusters)/args.n_clusters # Uniform distribution

        if args.cluster_balance == "kl":
            cluster_balance_loss = F.kl_div(P.log(), Q, reduction='batchmean')
        else:
            cluster_balance_loss = torch.linalg.norm(torch.sqrt(P) - torch.sqrt(Q))

        gating_err += args.eta*cluster_balance_loss
        gating_err.backward()
        optimizer.step()
        # print(f'Epoch {e+0:03}: | Gate Loss: {gating_err:.5f}')

    cluster_ids_train = torch.argmax(gate_vals, axis=1)
    print("Post Pre-training: ", np.bincount(cluster_ids_train))
    # wdfd_scores.append(calculate_WDFD(torch.FloatTensor(X_train), cluster_ids_train))

    # train Local Experts
    for e in range(N_EPOCHS):
        epoch_loss = 0
        epoch_auc = 0
        epoch_f1 = 0

        z_batch, x_bar, gate_vals = model(torch.Tensor(X_train).to(args.device))
        p_train = target_distribution(gate_vals.detach())

        model.train()
        for X_batch, y_batch, idx in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            z_batch, x_bar, gate_vals = model(X_batch)
            y_pred = torch.zeros((len(X_batch), args.n_classes))
            cluster_ids_train = torch.argmax(gate_vals, axis=1)
            cluster_centres = torch.matmul(z_batch.T, gate_vals)/z_train.shape[0]

            reconstr_loss = F.mse_loss(x_bar, X_batch)
            km_loss = torch.tensor(0.).to(args.device)
            train_loss = torch.tensor(0.).to(args.device)
            class_loss = torch.tensor(0.).to(args.device)
            cluster_balance_loss = torch.tensor(0.).to(args.device)

            for j in range(args.n_clusters):
                # cluster_ids = np.where(cluster_indices[idx] == j)[0] # not what is done in paper
                cluster_ids = range(len(X_batch))
                if args.sub_epochs == "True":
                    sub_epochs = min(10, 1 + int(e/5))
                else:
                    sub_epochs = 1
                z_cluster, y_cluster = z_batch[cluster_ids], y_batch[cluster_ids]
                # for _ in range(sub_epochs):
                preds_j = model.classifiers[j][0](z_cluster)
                optimizer_j = model.classifiers[j][1]
                optimizer_j.zero_grad()
                loss_j = nn.CrossEntropyLoss(reduction='mean')(preds_j, y_cluster)
                local_loss = torch.sum(gate_vals[cluster_ids,j]*loss_j)
                local_loss.backward(retain_graph=True)
                optimizer_j.step()

                preds_j = model.classifiers[j][0](z_cluster)
                loss_j = nn.CrossEntropyLoss(reduction='mean')(preds_j, y_cluster)
                y_pred[cluster_ids] += torch.reshape(gate_vals[cluster_ids, j], shape=(len(preds_j), 1)) * preds_j
                class_loss += torch.sum(gate_vals[cluster_ids,j]*loss_j)

            km_loss = F.kl_div(gate_vals.log(), p_train[idx], reduction='batchmean')
            # km_loss = torch.linalg.norm(torch.sqrt(gate_vals) - torch.sqrt(p_train[idx]))

            # P = torch.sum(torch.nn.Softmax(dim=1)(10*gate_vals), axis=0)
            P = torch.sum(gate_vals, axis=0)
            P = P/P.sum()
            Q = torch.ones(args.n_clusters)/args.n_clusters # Uniform distribution

            if args.cluster_balance == "kl":
                cluster_balance_loss = F.kl_div(P.log(), Q, reduction='batchmean')
            else:
                cluster_balance_loss = torch.linalg.norm(torch.sqrt(P) - torch.sqrt(Q))

            if args.alpha != 0:
                train_loss += args.alpha*reconstr_loss

            if args.beta != 0:
                train_loss += args.beta*km_loss

            if args.gamma != 0:
                train_loss += args.gamma*class_loss/z_batch.shape[0]

            if args.eta != 0:
                train_loss += args.eta*cluster_balance_loss

            optimizer.zero_grad()
            train_loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += train_loss

            train_metrics = performance_metrics(y_batch, y_pred.detach().numpy(), args.n_classes)
            f1  = train_metrics['f1_score']
            auc = train_metrics['auroc']
            auprc = train_metrics['auprc']
            minpse = train_metrics['minpse']
            acc = train_metrics['acc']

            epoch_auc += auc.item()
            epoch_f1 += f1.item()

        model.eval()
        z_val, x_bar, gate_vals = model(torch.FloatTensor(X_val).to(args.device))
        cluster_ids_val = torch.argmax(gate_vals, axis=1)
        print("Validation Cluster Situation: ", np.bincount(cluster_ids_val))

        val_pred = torch.zeros((len(X_val), args.n_classes))
        for j in range(args.n_clusters):
            model.classifiers[j][0].eval()
            preds_j = model.classifiers[j][0](z_val)
            loss_j = nn.CrossEntropyLoss(reduction='mean')(preds_j, torch.Tensor(y_val).type(torch.LongTensor))
            val_pred += torch.reshape(gate_vals[:,j], shape=(len(preds_j), 1)) * preds_j

        val_loss = nn.CrossEntropyLoss(reduction='mean')(val_pred, torch.tensor(y_val).to(device))
    
        val_metrics = performance_metrics(y_val, val_pred.detach().numpy(), args.n_classes)
        val_f1  = val_metrics['f1_score']
        val_auc = val_metrics['auroc']
        val_auprc = val_metrics['auprc']
        val_minpse = val_metrics['minpse']
        val_acc = val_metrics['acc']

        val_sil = silhouette_new(z_val.data.cpu().numpy(), cluster_ids_val.data.cpu().numpy(), metric='euclidean')

        if args.optimize == 'auc':
            opt = val_auc
        elif args.optimize == 'auprc':
            opt = val_auprc
        else:
            opt = -val_loss

        es([val_f1, opt], model)
        print(es.counter)
        print(f'Epoch {e+0:03}: | Train Loss: {epoch_loss/len(train_loader):.5f} | ',
        	f'Train F1: {epoch_f1/len(train_loader):.3f} | Train AUC: {epoch_auc/len(train_loader):.3f} | ',
        	f'Val F1: {val_f1:.3f} | Val Auc: {val_auc:.3f} | Val AUPRC: {val_auprc:.3f} | Val Sil: {val_sil:.3f} | Val Loss: {val_loss:.3f}')

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
    cluster_ids_test = torch.argmax(gate_test, axis=1)

    print("Test Cluster Situation: ", np.bincount(cluster_ids_test))
    test_sil = silhouette_new(z_test.data.cpu().numpy(), cluster_ids_test.data.cpu().numpy(), metric='euclidean')

    for j in range(args.n_clusters):
        model.classifiers[j][0].eval()
        preds_j = model.classifiers[j][0](z_test)
        loss_j = nn.CrossEntropyLoss(reduction='mean')(preds_j, torch.Tensor(y_test).type(torch.LongTensor))
        test_pred += torch.reshape(gate_test[:,j], shape=(len(preds_j), 1)) * preds_j

    test_loss = nn.CrossEntropyLoss(reduction='mean')(test_pred, torch.tensor(y_test).to(device))

    test_metrics = performance_metrics(y_test, test_pred.detach().numpy(), args.n_classes)
    test_f1  = test_metrics['f1_score']
    test_auc = test_metrics['auroc']
    test_auprc = test_metrics['auprc']
    test_minpse = test_metrics['minpse']
    test_acc = test_metrics['acc']

    # test_f1 = f1_score(np.argmax(test_pred.detach().numpy(), axis=1), y_test, average="macro")
    # test_auc = multi_class_auc(y_test, test_pred.detach().numpy(), args.n_classes)
    # test_auprc = multi_class_auprc(y_test, test_pred.detach().numpy(), args.n_classes)
    # test_acc = accuracy_score(np.argmax(test_pred.detach().numpy(), axis=1), y_test)
    es([val_f1, val_auprc], model)

    y_preds = np.argmax(test_pred.detach().numpy(), axis=1)
    print(confusion_matrix(y_test, y_preds))

    print(f'Epoch {e+0:03}: | Train Loss: {epoch_loss/len(train_loader):.5f} | ',
    	f'Train F1: {epoch_f1/len(train_loader):.3f} | Train AUC: {epoch_auc/len(train_loader):.3f} | ',
    	f'Test F1: {test_f1:.3f} | Test AUC: {test_auc:.3f} | Test AUPRC: {test_auprc:.3f} | ',
        f'Test Loss: {test_loss:.3f} | Test SIL: {test_sil:.3f}')

    print("\n####################################################################################\n")
    f1_scores.append(test_f1)
    auc_scores.append(test_auc)
    auprc_scores.append(test_auprc)
    minpse_scores.append(test_minpse)
    acc_scores.append(test_acc)
    nmi_scores.append(nmi_score(cluster_ids_test.data.cpu().numpy(), y_test))
    ari_scores.append(ari_score(cluster_ids_test.data.cpu().numpy(), y_test))
    test_cluster_nos.append(np.bincount(cluster_ids_test).shape[0])

    # sil_scores.append(silhouette_new(z_test.data.cpu().numpy(), cluster_ids_test.data.cpu().numpy(), metric='euclidean'))

    z_train, _, gate_vals = model(torch.FloatTensor(X_train))
    cluster_ids_train = torch.argmax(gate_vals, axis=1)
    sil_scores.append(silhouette_new(z_train.data.cpu().numpy(), cluster_ids_train.data.cpu().numpy(), metric='euclidean'))
    HTFD_scores.append(calculate_HTFD(torch.FloatTensor(X_train), cluster_ids_train))

enablePrint()
print("DMNN with ExpertNet training")
print("F1:", f1_scores)
print("AUC:", auc_scores)
print("AUPRC:", auprc_scores)
print("MINPSE:", minpse_scores)
print("SIL:", sil_scores)
print("Test Cluster Situation:", test_cluster_nos)
print("Sub Epochs:", args.sub_epochs)
print("Eta:", args.eta)

print("[Avg]\tDataset\tk\tF1\tAUC\tAUPRC\tMINPSE\tACC\tSIL\tHTFD\tWDFD")

print("\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format\
    (args.dataset, args.n_clusters, np.avg(f1_scores), np.avg(auc_scores),\
    np.avg(auprc_scores), np.avg(minpse_scores), np.avg(acc_scores),\
    np.avg(np.array(sil_scores)), np.avg(HTFD_scores), np.avg(wdfd_scores)))

print("[Std]\tF1\tAUC\tAUPRC\tMINPSE\tACC\tSIL\tHTFD\tWDFD")

print("\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format\
    (np.std(f1_scores), np.std(auc_scores),np.std(auprc_scores),\
    np.std(minpse_scores), np.std(acc_scores), np.std(np.array(sil_scores)),\
    np.std(HTFD_scores), np.std(wdfd_scores)))