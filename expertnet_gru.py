from __future__ import print_function, division
import copy
import torch
import argparse
import numpy as np
import os
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score,\
f1_score, roc_auc_score, average_precision_score, roc_curve, accuracy_score, matthews_corrcoef as mcc
from torch.utils.data import Subset
import pandas as pd
from scipy.spatial import distance_matrix

import argparse
import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.nn import Linear
from pytorchtools import EarlyStoppingEN_TS

import numbers
from sklearn.metrics import davies_bouldin_score as dbs, adjusted_rand_score as ari
from matplotlib import pyplot as plt
color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

from models import ExpertNet_GRU,  target_distribution, source_distribution
from utils import *
from ts_utils import *

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
parser.add_argument('--delta', default= 0.01, type=float) # Class equalization wt
parser.add_argument('--eta', default= 0.01, type=float) # Class seploss wt
parser.add_argument('--hidden_dims', default= [64, 32])
parser.add_argument('--n_z', default= 20, type=int)
parser.add_argument('--n_feats', default= 7, type=int)
parser.add_argument('--end_t', default= 24, type=int)
parser.add_argument('--n_clusters', default= 3, type=int)
parser.add_argument('--clustering', default= 'cac')
parser.add_argument('--n_classes', default= 2, type=int)

# Utility parameters
parser.add_argument('--device', default= 'cpu')
parser.add_argument('--verbose', default= 'False')
parser.add_argument('--cluster_analysis', default= 'False')
parser.add_argument('--log_interval', default= 10, type=int)
parser.add_argument('--pretrain_path', default= '/Users/shivin/Document/NUS/Research/CAC/CAC_DL/ExpertNet/pretrained_model')
# parser.add_argument('--pretrain_path', default= '/home/shivin/CAC_code/data')

parser = parser.parse_args()  
args = parameters(parser)
base_suffix = ""

for key in ['n_clusters', 'alpha', 'beta', 'gamma', 'delta', 'eta', 'attention']:
    print(key, args.__dict__[key])

base_suffix += args.dataset + "_"
base_suffix += str(args.n_clusters) + "_"
base_suffix += str(args.attention)

####################################################################################
####################################################################################
####################################################################################
################################### Initialiation ##################################
####################################################################################
####################################################################################
####################################################################################

f1_scores, auc_scores, auprc_scores, acc_scores, minpse_scores = [], [], [], [], [] #Inattentive test results
e_f1_scores, e_auc_scores, e_auprc_scores, e_acc_scores, e_minpse_scores = [], [], [], [], [] #Attentive test results
sil_scores, wdfd_scores, HTFD_scores, w_HTFD_scores = [], [], [], []

# to track the training loss as the model trains
train_losses, test_losses, e_test_losses, local_sum_test_losses = [], [], [], []
model_complexity = []

if args.ablation == "alpha":
    iter_array = alphas
    iteration_name = "Alpha"

elif args.ablation == "beta":
    iter_array = betas
    iteration_name = "Beta"

elif args.ablation == "gamma":
    iter_array = gammas
    iteration_name = "Gamma"

elif args.ablation == "delta":
    iter_array = deltas
    iteration_name = "Delta"

elif args.ablation == "k":
    iter_array = ks
    iteration_name = "K"

else:
    iter_array = range(args.n_runs)
    iteration_name = "Run"

for r in range(len(iter_array)):
    counter_batch = 0
    batch_loss = []
    model_batch_loss = []

    epoch_loss = []
    counter_batch = 0
    if args.verbose == 'False':
        blockPrint()

    print(iteration_name, ":", iter_array[r])

    if args.ablation == "beta":
        args.beta = iter_array[r]

    elif args.ablation == "gamma":
        args.gamma = iter_array[r]

    elif args.ablation == "delta":
        args.delta = iter_array[r]

    elif args.ablation == "k":
        args.n_clusters = iter_array[r]

    suffix = base_suffix + "_" + iteration_name + "_" + str(iter_array[r])

    train, val, test = get_ts_datasets(args, 0)
    X_train, X_train_len, y_train = train
    X_val, X_val_len, y_val = val
    X_test, X_test_len, y_test = test

    args.input_dim = X_train[0].shape[1]
    args.n_feats = args.input_dim
    args.end_t = 24
    pad_token = np.zeros(args.input_dim)
    device = args.device

    expert_layers = [args.n_z, 64, 32, 16, 8, args.n_classes]
    model = ExpertNet_GRU(
            expert_layers,
            args=args).to(args.device)

    model.train()
    optimizer = Adam(model.parameters(), lr=args.lr)

    print("Initializing Network Cluster Parameters")
    counter_batch = 0

    for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in enumerate(batch_iter(X_train, y_train, X_train_len, args.batch_size, shuffle=True)):
        # To implement
        # model.pretrain(train_loader, args.pretrain_path)
        counter_batch += len(x_batch)
        optimizer.zero_grad()
        x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_feats, args.end_t), dtype=torch.float32).to(device)
        x_batch = torch.nan_to_num(x_batch)

        _, _, hidden = model.ae(torch.Tensor(x_batch).to(args.device))
        original_cluster_centers, cluster_indices = kmeans2(hidden.data.cpu().numpy(), k=args.n_clusters, minit='++')
        model.cluster_layer.data += torch.tensor(original_cluster_centers).to(device)
        criterion = nn.CrossEntropyLoss(reduction='none')

    model.cluster_layer.data /= batch_idx


    print("Starting Training")
    
    for epoch in range(args.n_epochs):
        counter_batch = 0
        batch_loss = []
        model_batch_loss = []

        epoch_loss = []
        counter_batch = 0
        model.train()
        N_EPOCHS = args.n_epochs
        es = EarlyStoppingEN_TS(dataset=suffix)
        train_losses, e_train_losses = [], []

        epoch_loss = 0
        epoch_balance_loss = 0
        epoch_class_loss = 0
        epoch_km_loss = 0


        for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in enumerate(batch_iter(X_train, y_train, X_train_len, args.batch_size, shuffle=True)):
            # To implement
            # model.pretrain(train_loader, args.pretrain_path)
            counter_batch += len(x_batch)
            optimizer.zero_grad()
            x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_feats, args.end_t), dtype=torch.float32).to(device)
            
            if x_batch.shape[0] < args.n_clusters:
                continue

            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
            batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
            
            for i in range(len(batch_lens)):
                batch_lens[i] = min(batch_lens[i], args.end_t)

            masks = length_to_mask(batch_lens).unsqueeze(-1).float()
            x_batch = torch.nan_to_num(x_batch)
            
            ####################################################################################
            ####################################################################################
            ####################################################################################
            ################################## Clustering Step #################################
            ####################################################################################
            ####################################################################################
            ####################################################################################

            
            model.ae.train() # prep model for evaluation
            for j in range(model.n_clusters):
                model.classifiers[j][0].train()

            total_loss = 0
            x_batch = x_batch.to(device)

            X_latents, _, q_batch = model(x_batch)
            # reconstr_loss = F.mse_loss(x_bar, x_batch)
            reconstr_loss = 0 # to implement

            classifier_labels = np.zeros(len(x_batch))
            sub_epochs = min(10, 1 + int(epoch/5))

            if args.attention == False:
                classifier_labels = np.argmax(q_batch.detach().cpu().numpy(), axis=1)

            for _ in range(sub_epochs):
                # Choose classifier for a point probabilistically
                if args.attention == True:
                    for j in range(len(x_batch)):
                        classifier_labels[j] = np.random.choice(range(args.n_clusters), p = q_batch[j].detach().numpy())

                for k in range(args.n_clusters):
                    idx_cluster = np.where(classifier_labels == k)[0]
                    X_cluster = X_latents[idx_cluster]
                    y_cluster = y_batch[idx_cluster].type(torch.LongTensor)

                    classifier_k, optimizer_k = model.classifiers[k]
                    # Do not backprop the error to encoder
                    y_pred_cluster = classifier_k(X_cluster.detach())
                    cluster_loss = torch.mean(criterion(y_pred_cluster, y_cluster))
                    optimizer_k.zero_grad()
                    cluster_loss.backward(retain_graph=True)
                    optimizer_k.step()

            # Back propagate the error corresponding to last clustering
            class_loss = torch.tensor(0.).to(args.device)
            for k in range(args.n_clusters):
                idx_cluster = np.where(classifier_labels == k)[0]
                X_cluster = X_latents[idx_cluster]
                y_cluster = y_batch[idx_cluster].type(torch.LongTensor)

                classifier_k, optimizer_k = model.classifiers[k]
                y_pred_cluster = classifier_k(X_cluster)
                class_loss += torch.sum(q_batch[idx_cluster,k]*criterion(y_pred_cluster, y_cluster))

            class_loss /= len(X_latents)
            cluster_id = torch.argmax(q_batch, 1)
            delta_mu   = torch.zeros((args.n_clusters, args.latent_dim)).to(args.device)

            km_loss             = 0
            q_batch = source_distribution(X_latents, model.cluster_layer, alpha=model.alpha)
            p_batch = target_distribution(q_batch)

            P = torch.sum(torch.nn.Softmax(dim=1)(10*q_batch), axis=0)
            P = P/P.sum()
            Q = torch.ones(args.n_clusters)/args.n_clusters # Uniform distribution

            if args.cluster_balance == "kl":
                cluster_balance_loss = F.kl_div(P.log(), Q, reduction='batchmean')
            else:
                cluster_balance_loss = torch.linalg.norm(torch.sqrt(P) - torch.sqrt(Q))

            # km_loss = F.kl_div(q_batch.log(), p_train[idx_batch], reduction='batchmean')
            km_loss = F.kl_div(q_batch.log(), p_batch, reduction='batchmean')

            loss = reconstr_loss
            if args.beta != 0:
                loss += args.beta*km_loss
            if args.gamma != 0:
                loss += args.gamma*class_loss
            if args.delta != 0:
                loss += args.delta*cluster_balance_loss

            epoch_loss += loss
            epoch_class_loss += class_loss
            epoch_balance_loss += cluster_balance_loss
            epoch_km_loss += km_loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        print('Epoch: {:02d} | Batch: {:02d} | Batch KM Loss: {:.3f} | Total Loss: {:.3f} | Classification Loss: {:.3f} |\
        Cluster Balance Loss: {:.3f}'.format(epoch, batch_idx, epoch_km_loss, epoch_loss, epoch_class_loss, epoch_balance_loss))
        train_losses.append([np.round(epoch_loss.item(),3), np.round(epoch_class_loss.item(),3)])

        ######################################################
        ######################################################
        ################### Validation #######################
        ######################################################
        ######################################################

        y_true = []
        y_pred = []
        z_val = []
        cluster_ids_val = np.array([])
        with torch.no_grad():
            model.eval()
            model.ae.eval() # prep model for evaluation
            for j in range(model.n_clusters):
                model.classifiers[j][0].eval()

            for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in enumerate(batch_iter(X_val, y_val, X_val_len, args.batch_size, shuffle=True)):
                optimizer.zero_grad()
                x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_feats, args.end_t), dtype=torch.float32).to(device)
                
                if x_batch.shape[0] < args.n_clusters:
                    continue

                y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
                batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
                
                for i in range(len(batch_lens)):
                    batch_lens[i] = min(batch_lens[i], args.end_t)

                masks = length_to_mask(batch_lens).unsqueeze(-1).float()
                x_batch = torch.nan_to_num(x_batch)

                # Evaluate model on Validation dataset
                q_batch, z_batch = model(torch.FloatTensor(x_batch).to(args.device), output="latent")
                cluster_ids = torch.argmax(q_batch, axis=1)
                preds = torch.zeros((len(z_batch), args.n_classes))

                z_val.append(z_batch.detach().numpy())
                cluster_ids_val = np.hstack([cluster_ids_val, cluster_ids.detach().numpy()])

                # Weighted predictions
                if args.attention == False:
                    for j in range(model.n_clusters):
                        cluster_id = np.where(cluster_ids == j)[0]
                        X_cluster = z_batch[cluster_id]
                        cluster_preds_val = model.classifiers[j][0](X_cluster)
                        preds[cluster_id,:] = cluster_preds_val

                else:
                    for j in range(model.n_clusters):
                        X_cluster = z_batch
                        cluster_preds = model.classifiers[j][0](X_cluster)
                        for c in range(args.n_classes):
                            preds[:,c] += q_batch[:,j]*cluster_preds[:,c]

                y_pred += list(preds.cpu().detach().numpy())
                y_true += list(y_batch.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        z_val = np.concatenate(z_val, axis=0)


        # Classification Matrics
        val_metrics = performance_metrics(y_true, y_pred, args.n_classes)
        val_f1  = val_metrics['f1_score']
        val_auc = val_metrics['auroc']
        val_auprc = val_metrics['auprc']
        val_minpse = val_metrics['minpse']
        
        # Clustering Metrics
        val_sil = silhouette_new(z_val, np.array(cluster_ids_val), metric='euclidean')
        val_feature_diff, val_WDFD = 0, 0
        # val_feature_diff = calculate_HTFD(X_val, cluster_ids)
        # val_WDFD = calculate_WDFD(X_val, cluster_ids)
        val_feature_diff = 0
        val_WDFD = 0
        val_loss = torch.mean(criterion(torch.Tensor(y_pred), torch.Tensor(y_true).type(torch.LongTensor)))

        epoch_len = len(str(N_EPOCHS))

        print_msg = (f'\n[{epoch:>{epoch_len}}/{N_EPOCHS:>{epoch_len}}] ' +
                     # f'train_loss: {train_loss:.3f} ' +
                     f'valid_loss: {val_loss:.3f} '  +
                     f'valid_F1: {val_f1:.3f} '  +
                     f'valid_AUC: {val_auc:.3f} ' + 
                     f'valid_AUPRC: {val_auprc:.3f} ' + 
                     f'valid_MINPSE: {val_minpse:.3f} ' + 
                     f'valid_Feature_p: {val_feature_diff:.3f} ' + 
                     f'valid_WDFD: {val_WDFD:.3f} ' + 
                     f'valid_Silhouette: {val_sil:.3f}')

        print(print_msg)
        print("\n")

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        es([val_f1, val_auprc], model)
        if es.early_stop == True:
            break

    ####################################################################################
    ####################################################################################
    ####################################################################################
    ################################### Local Training #################################
    ####################################################################################
    ####################################################################################
    ####################################################################################

        
    print("\n####################################################################################\n")
    print("Training Local Networks")
    model = es.load_checkpoint(model)
    es = EarlyStoppingEN_TS(dataset=suffix)

    for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in enumerate(batch_iter(X_train, y_train, X_train_len, args.batch_size, shuffle=True)):
        # To implement
        # model.pretrain(train_loader, args.pretrain_path)
        counter_batch += len(x_batch)
        optimizer.zero_grad()
        x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_feats, args.end_t), dtype=torch.float32).to(device)
        
        if x_batch.shape[0] < args.n_clusters:
            continue

        y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
        batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
        
        for i in range(len(batch_lens)):
            batch_lens[i] = min(batch_lens[i], args.end_t)

        masks = length_to_mask(batch_lens).unsqueeze(-1).float()
        x_batch = torch.nan_to_num(x_batch)
        z_batch, _, q_batch = model(x_batch)
        cluster_id_batch = torch.argmax(q_batch, axis=1)

        # Post clustering training
        for e in range(args.n_epochs):
            epoch_loss = 0
            epoch_acc = 0
            epoch_f1 = 0
            acc = 0

            # Ready local classifiers for training
            for j in range(args.n_clusters):
                model.classifiers[j][0].train()

            classifier_labels = np.zeros(len(z_batch))

            # Choose classifier for a point probabilistically            
            if args.attention == True:
                for j in range(len(z_batch)):
                    classifier_labels[j] = np.random.choice(range(args.n_clusters), p = q_batch[j].detach().numpy())
            else:
                classifier_labels = torch.argmax(q_batch, axis=1).data.cpu().numpy()

            for k in range(args.n_clusters):
                idx_cluster = np.where(classifier_labels == k)[0]
                X_cluster = z_batch[idx_cluster]
                y_cluster = y_batch[idx_cluster].type(torch.LongTensor)

                classifier_k, optimizer_k = model.classifiers[k]

                # Do not backprop the error to encoder
                y_pred_cluster = classifier_k(X_cluster.detach())
                cluster_loss = torch.mean(criterion(y_pred_cluster, y_cluster))
                optimizer_k.zero_grad()
                cluster_loss.backward(retain_graph=True)
                optimizer_k.step()

            # Ready local classifiers for evaluation
            for j in range(model.n_clusters):
                model.classifiers[j][0].eval()

            train_preds = torch.zeros((len(z_batch), args.n_classes))
            train_loss = 0

            # Weighted predictions
            q_batch, z_batch = model(torch.FloatTensor(x_batch).to(args.device), output="latent")
            cluster_ids_batch = torch.argmax(q_batch, axis=1)
            
            for j in range(model.n_clusters):
                cluster_id = np.where(cluster_ids_batch == j)[0]
                X_cluster = z_batch
                y_cluster = torch.Tensor(y_batch[cluster_id]).type(torch.LongTensor)

                # Ensemble train loss
                cluster_preds = model.classifiers[j][0](X_cluster)
                for c in range(args.n_classes):
                    train_preds[:,c] += q_batch[:,j]*cluster_preds[:,c]

                X_cluster = z_batch[cluster_id]
                cluster_preds = model.classifiers[j][0](X_cluster)
                train_loss += torch.sum(q_batch[cluster_id,j]*criterion(cluster_preds, y_cluster))

            train_loss /= len(z_batch)
        
        print_msg = (f'\n[{epoch:>{epoch_len}}/{N_EPOCHS:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.3f} ')
        print(print_msg)
        print("\n")


        ## Local training validation performance
        y_true = []
        y_pred = []
        z_val = []
        cluster_ids_val = np.array([])

        for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in enumerate(batch_iter(X_val, y_val, X_val_len, args.batch_size, shuffle=True)):
            optimizer.zero_grad()
            x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_feats, args.end_t), dtype=torch.float32).to(device)
            
            if x_batch.shape[0] < args.n_clusters:
                continue

            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
            batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
            
            for i in range(len(batch_lens)):
                batch_lens[i] = min(batch_lens[i], args.end_t)

            masks = length_to_mask(batch_lens).unsqueeze(-1).float()
            x_batch = torch.nan_to_num(x_batch)

            # Evaluate model on Validation dataset
            q_batch, z_batch = model(torch.FloatTensor(x_batch).to(args.device), output="latent")
            cluster_ids = torch.argmax(q_batch, axis=1)

            preds = torch.zeros((len(z_batch), args.n_classes))

            # Weighted predictions
            for j in range(model.n_clusters):
                cluster_id = np.where(cluster_ids == j)[0]
                X_cluster = z_batch
                cluster_preds = model.classifiers[j][0](X_cluster)
                for c in range(args.n_classes):
                    preds[:,c] += q_batch[:,j]*cluster_preds[:,c]

            y_pred += list(preds.cpu().detach().numpy())
            y_true += list(y_batch.cpu().numpy())
            z_val.append(z_batch.detach().numpy())
            cluster_ids_val = np.hstack([cluster_ids_val, cluster_ids.detach().numpy()])


        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        z_val = np.concatenate(z_val, axis=0)

        val_metrics = performance_metrics(y_true, y_pred, args.n_classes)
        val_f1  = val_metrics['f1_score']
        val_auc = val_metrics['auroc']
        val_auprc = val_metrics['auprc']
        val_minpse = val_metrics['minpse']
        
        # Clustering Metrics
        val_sil = silhouette_new(z_val, np.array(cluster_ids_val), metric='euclidean')
        val_feature_diff, val_WDFD = 0, 0
        # val_feature_diff = calculate_HTFD(X_val, cluster_ids)
        # val_WDFD = calculate_WDFD(X_val, cluster_ids)
        val_feature_diff = 0
        val_WDFD = 0
        val_loss = torch.mean(criterion(torch.Tensor(y_pred), torch.Tensor(y_true).type(torch.LongTensor)))
        
        print_msg = (f'\n[{epoch:>{epoch_len}}/{N_EPOCHS:>{epoch_len}}] ' +
                     f'valid_loss: {val_loss:.3f} '  +
                     f'valid_F1: {val_f1:.3f} '  +
                     f'valid_AUC: {val_auc:.3f} ' +
                     f'valid_AUPRC: {val_auprc:.3f} ' +
                     f'valid_MINPSE: {val_minpse:.3f} ' +
                     f'valid_Sil: {val_sil:.3f}')
        
        print(print_msg)
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        es([val_f1, val_auprc], model)
        if es.early_stop == True:
            # train_losses.append(train_loss.item())
            sil_scores.append(silhouette_new(z_val.data.cpu().numpy(), cluster_ids_val.data.cpu().numpy(), metric='euclidean'))
            # HTFD_scores.append(calculate_HTFD(X_train, cluster_ids_train))
            # wdfd_scores.append(calculate_WDFD(X_train, cluster_ids_train))
            break


    ####################################################################################
    ####################################################################################
    ####################################################################################
    ################################### Testing Perf. ##################################
    ####################################################################################
    ####################################################################################
    ####################################################################################

    print("\n####################################################################################\n")
    print("Evaluating Test Data with k = ", args.n_clusters, " Attention = ", args.attention)

    # Load best model trained from local training phase
    model = es.load_checkpoint(model)
    y_true = []
    y_pred = []
    z_test = []
    cluster_ids_test = np.array([])


    for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in enumerate(batch_iter(X_test, y_test, X_test_len, args.batch_size, shuffle=True)):
        optimizer.zero_grad()
        x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_feats, args.end_t), dtype=torch.float32).to(device)
        
        if x_batch.shape[0] < args.n_clusters:
            continue

        y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
        batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
        
        for i in range(len(batch_lens)):
            batch_lens[i] = min(batch_lens[i], args.end_t)

        masks = length_to_mask(batch_lens).unsqueeze(-1).float()
        x_batch = torch.nan_to_num(x_batch)

        # Evaluate model on Validation dataset
        q_batch, z_batch = model(torch.FloatTensor(x_batch).to(args.device), output="latent")
        cluster_ids = torch.argmax(q_batch, axis=1)

        preds = torch.zeros((len(z_batch), args.n_classes))

        # Weighted predictions
        for j in range(model.n_clusters):
            cluster_id = np.where(cluster_ids == j)[0]
            X_cluster = z_batch
            cluster_preds = model.classifiers[j][0](X_cluster)
            for c in range(args.n_classes):
                preds[:,c] += q_batch[:,j]*cluster_preds[:,c]

        y_pred += list(preds.cpu().detach().numpy())
        y_true += list(y_batch.cpu().numpy())
        z_test.append(z_batch.detach().numpy())
        cluster_ids_test = np.hstack([cluster_ids_test, cluster_ids.detach().numpy()])


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    z_test = np.concatenate(z_test, axis=0)

    test_metrics = performance_metrics(y_true, y_pred, args.n_classes)
    test_f1  = test_metrics['f1_score']
    test_auc = test_metrics['auroc']
    test_auprc = test_metrics['auprc']
    test_minpse = test_metrics['minpse']
    
    # Clustering Metrics
    test_sil = silhouette_new(z_test, np.array(cluster_ids_test), metric='euclidean')
    test_feature_diff, test_WDFD = 0, 0
    # test_feature_diff = calculate_HTFD(X_test, cluster_ids)
    # test_WDFD = calculate_WDFD(X_test, cluster_ids)
    test_feature_diff = 0
    test_WDFD = 0
    test_loss = torch.mean(criterion(torch.Tensor(y_pred), torch.Tensor(y_true).type(torch.LongTensor)))

    test_metrics = performance_metrics(y_batch, preds.detach().numpy(), args.n_classes)
    test_f1  = test_metrics['f1_score']
    test_auc = test_metrics['auroc']
    test_auprc = test_metrics['auprc']
    test_minpse = test_metrics['minpse']
    test_acc = test_metrics['acc']

    test_sil = silhouette_new(z_test.data, cluster_ids_test, metric='euclidean')
    test_loss = torch.mean(criterion(preds, torch.Tensor(y_batch).type(torch.LongTensor)))
    epoch_len = len(str(N_EPOCHS))

    # print_msg = (f'\n[{epoch:>{epoch_len}}/{N_EPOCHS:>{epoch_len}}] ' +
    #              f'test_loss: {test_loss:.3f} '  +
    #              f'test_F1: {test_f1:.3f} '  +
    #              f'test_AUC: {test_auc:.3f} ' +
    #              f'test_AUPRC: {test_auprc:.3f} ' +
    #              f'test_MINPSE: {test_minpse:.3f} ' +
    #              f'test_Sil: {test_sil:.3f}')
    
    # print(print_msg)

    print("Run #{}".format(r))
    print('Loss Metrics - Test Loss {:.3f}, E-Test Loss {:.3f}'.format(test_loss, test_loss))

    # print('Clustering Metrics - Acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi),\
          # ', ari {:.4f}, HTFD {:.3f}'.format(ari, e_test_HTFD))

    print('Classification Metrics - E-Test F1 {:.3f}, E-Test AUC {:.3f}, E-Test AUPRC {:.3f}, E-Test MIN_PSE {:.3f}'.format\
        (test_f1, test_auc, test_auprc, test_minpse))

    print("\n")

    f1_scores.append(test_f1)
    auc_scores.append(test_auc)
    auprc_scores.append(test_auprc)
    minpse_scores.append(test_minpse)
    acc_scores.append(test_acc)

    ####################################################################################
    ####################################################################################
    ####################################################################################
    ################################### Feature Imp. ###################################
    ####################################################################################
    ####################################################################################
    ####################################################################################

    # regs = [GradientBoostingClassifier(random_state=0) for _ in range(args.n_clusters)]
    # qs, z_train = model(torch.FloatTensor(X_train).to(args.device), output="latent")
    # q_train = qs[0]
    # cluster_ids = torch.argmax(q_train, axis=1)
    # train_preds = torch.zeros((len(z_train), args.n_classes))
    # feature_importances = np.zeros((args.n_clusters, args.input_dim))

    # # Weighted predictions... should be wihout attention only
    # for j in range(model.n_clusters):
    #     cluster_id = torch.where(cluster_ids == j)[0]
    #     X_cluster = z_train[cluster_id]
    #     cluster_preds = model.classifiers[j][0](X_cluster)
    #     # print(q_test, cluster_preds[:,0])
    #     train_preds[cluster_id,:] = cluster_preds
    #     # y_cluster = np.argmax(train_preds[cluster_id].detach().numpy(), axis=1)
    #     y_cluster = y_train[cluster_id]

    #     # Train the local regressors on the data embeddings
    #     # Some test data might not belong to any cluster
    #     if len(cluster_id) > 0:
    #         regs[j].fit(X_train[cluster_id], y_cluster)
    #         best_features = np.argsort(regs[j].feature_importances_)[::-1][:10]
    #         feature_importances[j,:] = regs[j].feature_importances_
    #         print("Cluster # ", j, "sized: ", len(cluster_id), "label distr: ", sum(y_cluster)/len(y_cluster))
    #         print(list(zip(column_names[best_features], np.round(regs[j].feature_importances_[best_features], 3))))
    #         print("=========================\n")

    # # Testing performance of downstream classifier on cluster embeddings
    # qs, z_test = model(torch.FloatTensor(X_test).to(args.device), output="latent")
    # q_test = qs[0]
    # cluster_ids_test = torch.argmax(q_test, axis=1)
    # test_preds = torch.zeros((len(z_test), args.n_classes))
    # y_pred = np.zeros((len(z_test), args.n_classes))

    # for j in range(model.n_clusters):
    #     cluster_id = torch.where(cluster_ids_test == j)[0]
    #     X_cluster = X_test[cluster_id]

    #     # Some test data might not belong to any cluster
    #     if len(cluster_id) > 0:
    #         y_pred[cluster_id] = regs[j].predict_proba(X_cluster)

    # feature_diff = 0
    # cntr = 0
    # for i in range(args.n_clusters):
    #     for j in range(args.n_clusters):
    #         if i > j:
    #             ci = torch.where(cluster_ids == i)[0]
    #             cj = torch.where(cluster_ids == j)[0]
    #             Xi = X_train[ci]
    #             Xj = X_train[cj]
    #             feature_diff += 100*sum(feature_importances[i]*feature_importances[j]*(ttest_ind(Xi, Xj, axis=0)[1] < 0.05))/args.input_dim
    #             # print("Cluster [{}, {}] p-value: ".format(i,j), feature_diff)
    #             cntr += 1

    # if cntr == 0:
    #     w_HTFD_scores.append(0)
    # else:
    #     print("Average Feature Difference: ", feature_diff/cntr)
    #     w_HTFD_scores.append(feature_diff/cntr)


print("\n")
print("Experiment ", iteration_name)
print(iter_array)
print("Test F1: ", f1_scores)
print("Test AUC: ", auc_scores)
print("Test AUPRC: ", auprc_scores)
print("Test MINPSE: ", minpse_scores)

print("Sil scores: ", sil_scores)
print("HTFD: ", HTFD_scores)
print("WDFD: ", wdfd_scores)

# print("Train Loss: ", train_losses)
# print("E-Train Loss: ", e_train_losses)

# print("Test Loss: ", test_losses)

enablePrint()

print("[Avg]\tDataset\tk\tE-F1\tE-AUC\tE-AUPRC\tE-MINPSE\tE-ACC")
print("\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(args.dataset, args.n_clusters,\
    np.average(f1_scores), np.average(auc_scores), np.average(auprc_scores), np.average(minpse_scores), np.average(acc_scores)))

print("[Std]\tE-F1\tE-AUC\tE-AUPRC\tE-MINPSE\tE-ACC")
print("\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format\
    (np.std(f1_scores), np.std(auc_scores), np.std(auprc_scores), np.std(minpse_scores), np.std(acc_scores)))

print('[Avg]\tSIL\tHTFD\tWDFD\tW-HTFD')
print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(np.average(sil_scores),\
    np.average(HTFD_scores), np.average(wdfd_scores), np.average(w_HTFD_scores)))

print('[Std]\tSIL\tHTFD\tWDFD\tW-HTFD')
print("\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(np.std(sil_scores),\
    np.std(HTFD_scores), np.std(wdfd_scores), np.std(w_HTFD_scores)))

# print("F1\tAUC\tAUPRC\tACC")

# print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format\
#     (np.average(f1_scores), np.average(auc_scores), np.average(auprc_scores), np.average(acc_scores)))

# print("Train Loss\tE-Train Loss\tTest Loss\tE-Test Loss")

# print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format\
#     (np.average(train_losses), np.average(e_train_losses),\
#     np.average(test_losses), np.average(e_test_losses)))

if args.cluster_analysis == "True":
    WDFD_Cluster_Analysis(torch.Tensor(X_train), cluster_ids_train, column_names)
    HTFD_Cluster_Analysis(torch.Tensor(X_train), cluster_ids_train, column_names)
    HTFD_Single_Cluster_Analysis(X_train, y_train, cluster_ids_train, column_names)