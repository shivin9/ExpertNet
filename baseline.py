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
from scipy.cluster.vq import vq, kmeans2
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
from pytorchtools import EarlyStopping

import numbers
from sklearn.metrics import davies_bouldin_score as dbs, adjusted_rand_score as ari
from matplotlib import pyplot as plt
color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

from models import NNClassifier, NNClassifierBase, pretrain_ae
from utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default= 'creditcard')
parser.add_argument('--input_dim', default= '-1')
parser.add_argument('--n_features', default= '-1')

# Training parameters
parser.add_argument('--lr_enc', default= 0.002, type=float)
parser.add_argument('--lr_exp', default= 0.002, type=float)
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
parser.add_argument('--optimize', default= 'auprc')

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
parser.add_argument('--expt', default= 'ExpertNet')
parser.add_argument('--cluster_analysis', default= 'False')
parser.add_argument('--pretrain_path', default= '/Users/shivin/Document/NUS/Research/CAC/CAC_DL/ExpertNet/pretrained_model/Base')


parser = parser.parse_args()
args = parameters(parser)

####################################################################################
####################################################################################
####################################################################################
###################################### Training ####################################
####################################################################################
####################################################################################
####################################################################################

f1_scores, auc_scores, auprc_scores, minpse_scores, acc_scores = [], [], [], [], []
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

    if args.expt == 'ExpertNet':
        ae_layers = [128, 64, args.n_z, 64, 128]
        expert_layers = [args.n_z, 128, 64, 32, 16, args.n_classes]

    else:
        ae_layers = [64, args.n_z, 64]
        expert_layers = [args.n_z, 30, args.n_classes]

    model = NNClassifier(ae_layers, expert_layers, args).to(args.device)
    model.pretrain(train_loader, args.pretrain_path)
    
    # Single Big NN    
    # layers = [args.input_dim, 512, 256, 64, args.n_classes]
    # layers = [args.input_dim, 128, 64, 32, 16, args.n_classes]
    # model = NNClassifierBase(args, input_dim=args.input_dim, layers=layers)

    # For DeepCAC baselines. Single Big NN
    # layers = [args.input_dim, 64, 32, 30, args.n_classes]
    # model = NNClassifierBase(args, input_dim=args.input_dim, layers=layers)

    device = args.device

    N_EPOCHS = args.n_epochs
    es = EarlyStopping(dataset=args.dataset)

    for e in range(1, N_EPOCHS):
        epoch_loss = 0
        epoch_auc = 0
        epoch_f1 = 0
        auc = 0
        model.train()
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred, train_loss = model.fit(X_batch, y_batch)
            epoch_loss += train_loss

            f1 = f1_score(np.argmax(y_pred, axis=1), y_batch.detach().numpy(), average="macro")
            auc = multi_class_auc(y_batch, y_pred, args.n_classes)
            epoch_auc += auc.item()
            epoch_f1 += f1.item()

        model.classifier.eval()
        _, last_vals, val_preds = model(torch.FloatTensor(np.array(X_val)).to(args.device))
        val_loss = nn.CrossEntropyLoss(reduction='mean')(val_preds, torch.tensor(y_val).to(device))
        val_metrics = performance_metrics(y_val, val_preds.detach().numpy(), args.n_classes)

        val_f1  = val_metrics['f1_score']
        val_auc = val_metrics['auroc']
        val_auprc = val_metrics['auprc']
        val_minpse = val_metrics['minpse']
        val_acc = val_metrics['acc']

        if args.optimize == 'auc':
            opt = val_auc
        elif args.optimize == 'auprc':
            opt = val_auprc
        else:
            opt = -val_loss

        es([val_f1, opt], model)

        # original_cluster_centers, cluster_indices = kmeans2(last_vals.data.cpu().numpy(), k=args.n_clusters, minit='++')
        # plot(model, torch.FloatTensor(last_vals).to(args.device), y_val, args, labels=cluster_indices, epoch=e)
        # idx = range(int(0.2*len(X_val)))

        # plot_data(torch.FloatTensor(last_vals)[idx].to(args.device), y_val[idx], cluster_indices[idx], args, e)
        
        print(f'Epoch {e+0:03}: | Train Loss: {epoch_loss/len(train_loader):.5f} | ',
        	f'Train F1: {epoch_f1/len(train_loader):.3f} | Train Auc: {epoch_auc/len(train_loader):.3f} | ',
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
    model.classifier.eval()
    _, last_test, test_preds = model(torch.FloatTensor(np.array(X_test)).to(args.device))
    test_loss = nn.CrossEntropyLoss(reduction='mean')(test_preds, torch.tensor(y_test).to(device))

    test_f1 = f1_score(np.argmax(test_preds.detach().numpy(), axis=1), y_test, average="macro")
    test_auc = multi_class_auc(y_test, test_preds.detach().numpy(), args.n_classes)
    test_auprc = multi_class_auprc(y_test, test_preds.detach().numpy(), args.n_classes)
    test_acc = accuracy_score(np.argmax(test_preds.detach().numpy(), axis=1), y_test)

    test_metrics = performance_metrics(y_test, test_preds.detach().numpy(), args.n_classes)
    test_f1  = test_metrics['f1_score']
    test_auc = test_metrics['auroc']
    test_auprc = test_metrics['auprc']
    test_minpse = test_metrics['minpse']
    test_acc = test_metrics['acc']

    y_preds = np.argmax(test_preds.detach().numpy(), axis=1)

    print(f'Epoch {e+0:03}: | Train Loss: {epoch_loss/len(train_loader):.5f} | ',
    	f'Train F1: {epoch_f1/len(train_loader):.3f} | Train Auc: {epoch_auc/len(train_loader):.3f}| ',
    	f'Test F1: {test_f1:.3f} | Test Auc: {test_auc:.3f} | Test Loss: {test_loss:.3f}')

    print("\n####################################################################################\n")
    f1_scores.append(test_f1)
    auc_scores.append(test_auc)
    auprc_scores.append(test_auprc)
    minpse_scores.append(test_minpse)
    acc_scores.append(test_acc)

enablePrint()
print("F1:", f1_scores)
print("AUC:", auc_scores)
print("AUPRC:", auprc_scores)
print("MINPSE:", minpse_scores)
print("ACC:", acc_scores)

print("[Avg]\tDataset\tk\tF1\tAUC\tAUPRC\tMINPSE\tACC")
print("\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(args.dataset, args.n_clusters,\
    np.avg(f1_scores), np.avg(auc_scores), np.avg(auprc_scores), np.avg(minpse_scores), np.avg(acc_scores)))

print("[Std]\tDataset\tk\tF1\tAUC\tAUPRC\tMINPSE\tACC")
print("\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format\
    (np.std(f1_scores), np.std(auc_scores), np.std(auprc_scores), np.std(minpse_scores), np.std(acc_scores)))
