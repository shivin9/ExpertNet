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
parser.add_argument('--input_dim', default= -1, type=int)
parser.add_argument('--n_features', default= -1, type=int)

# Training parameters
parser.add_argument('--lr_enc', default= 0.002, type=float)
parser.add_argument('--lr_exp', default= 0.002, type=float)
parser.add_argument('--alpha', default= 1, type=float)
parser.add_argument('--wd', default= 5e-4, type=float)
parser.add_argument('--batch_size', default= 512, type=int)
parser.add_argument('--n_epochs', default= 10, type=int)
parser.add_argument('--n_runs', default= 3, type=int)
parser.add_argument('--pre_epoch', default= 40, type=int)
parser.add_argument('--pretrain', default= True, type=bool)
parser.add_argument("--load_ae",  default=False, type=bool)
parser.add_argument("--classifier", default="LR")
parser.add_argument("--tol", default=0.01, type=float)
parser.add_argument("--attention", default=11, type=int)
parser.add_argument('--ablation', default='None')
parser.add_argument('--cluster_balance', default='hellinger')
parser.add_argument('--end_t', default= 24, type=int)

# Model parameters
parser.add_argument('--lamda', default= 1, type=float)
parser.add_argument('--beta', default= 0.5, type=float) # KM loss wt
parser.add_argument('--gamma', default= 1.0, type=float) # Classification loss wt
parser.add_argument('--delta', default= 0.01, type=float) # Class equalization wt
parser.add_argument('--eta', default= 0.01, type=float) # Class seploss wt
parser.add_argument('--hidden_dims', default= [64, 32])
parser.add_argument('--n_z', default= 20, type=int)
parser.add_argument('--n_clusters', default= 3, type=int)
parser.add_argument('--clustering', default= 'cac')
parser.add_argument('--n_classes', default= 2, type=int)
parser.add_argument('--optimize', default= 'auprc')

# Utility parameters
parser.add_argument('--device', default= 'cpu')
parser.add_argument('--verbose', default= 'False')
parser.add_argument('--plot', default= 'False')
parser.add_argument('--expt', default= 'ExpertNet')
parser.add_argument('--cluster_analysis', default= 'False')
parser.add_argument('--log_interval', default= 10, type=int)
parser.add_argument('--pretrain_path', default= '/Users/shivin/Document/NUS/Research/CAC/CAC_DL/ExpertNet/pretrained_model/EN_TS')
# parser.add_argument('--pretrain_path', default= '/home/shivin/CAC_DL/pretrained_model/EN_TS')

parser = parser.parse_args()
args = parameters(parser)
base_suffix = ""

for key in ['n_clusters', 'alpha', 'beta', 'gamma', 'delta', 'eta', 'attention']:
    print(key, args.__dict__[key])

base_suffix += args.dataset + "_"
base_suffix += str(args.n_clusters) + "_"
base_suffix += str(args.attention)

if args.attention == 0:
    attention_train = 0
    attention_test = 1

elif args.attention == 1:
    attention_train = 0
    attention_test = 1

elif args.attention == 10:
    attention_train = 1
    attention_test = 0

elif args.attention == 11:
    attention_train = 1
    attention_test = 1


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

    train, val, test, scale = get_ts_datasets(args, r_state=r)
    X_train, X_train_len, y_train = train
    X_val, X_val_len, y_val = val
    X_test, X_test_len, y_test = test
    pad_token = np.zeros(args.input_dim)
    device = args.device

    expert_layers = [args.n_z, 128, 64, 32, 16, args.n_classes]
    model = ExpertNet_GRU(
            expert_layers,
            args.lr_enc,
            args.lr_exp,
            args=args).to(device)

    model.train()

    print("Initializing Network Cluster Parameters")
    counter_batch = 0
    model.pretrain(train, args.pretrain_path)
    
    print("Input Dim:", args.input_dim, "#Features:", args.n_features)

    # Initializing cluster centers, can't pass all points in one go
    for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in enumerate(batch_iter(X_train, y_train, X_train_len, args.batch_size)):
        # To implement
        counter_batch += len(x_batch)
        model.optimizer.zero_grad()
        x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_features, args.end_t), dtype=torch.float32).to(device)
        x_batch = torch.nan_to_num(x_batch).to(device)

        _, _, hidden = model.ae(x_batch)
        original_cluster_centers, cluster_indices = kmeans2(hidden.data.cpu().numpy(), k=args.n_clusters, minit='++')
        model.cluster_layer += torch.tensor(original_cluster_centers).to(device)

    model.cluster_layer.data /= batch_idx
    criterion = nn.CrossEntropyLoss(reduction='none')

    print("Starting Training")
    es = EarlyStoppingEN_TS(dataset=suffix)
    p_train = torch.zeros(len(X_train), args.n_clusters).to(args.device)

    for epoch in range(args.n_epochs):
        counter_batch = 0
        batch_loss = []
        model_batch_loss = []
        epoch_loss = []
        q_train = []
        x_train = []

        counter_batch = 0
        model.train()
        N_EPOCHS = args.n_epochs
        train_losses, e_train_losses = [], []

        epoch_loss = 0
        epoch_balance_loss = 0
        epoch_class_loss = 0
        epoch_km_loss = 0
        n_batches = 0

        for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in enumerate(batch_iter(X_train, y_train, X_train_len, args.batch_size)):
            # To implement
            counter_batch += len(x_batch)
            model.optimizer.zero_grad()
            x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_features, args.end_t), dtype=torch.float32).to(device)
            
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
            y_batch = y_batch.type(torch.LongTensor).to(model.device)

            X_latents, x_bar, q_batch = model.encoder_forward(x_batch, output="decoded")

            q_train.append(q_batch.detach().cpu().numpy())
            x_train.append(x_batch.detach().cpu().numpy())

            reconstr_loss = F.mse_loss(x_bar, x_batch)
            classifier_labels = np.zeros(len(x_batch))
            sub_epochs = min(10, 1 + int(epoch/5))

            for _ in range(sub_epochs):
                model.expert_forward(x_batch, y_batch, backprop_enc=False, backprop_local=True, attention=attention_train)

            _, class_loss = model.expert_forward(x_batch, y_batch, backprop_enc=True, backprop_local=False, attention=attention_train)

            class_loss /= len(X_latents)
            cluster_id = torch.argmax(q_batch, 1)
            delta_mu   = torch.zeros((args.n_clusters, args.latent_dim)).to(args.device)

            km_loss = 0
            q_batch = source_distribution(X_latents, model.cluster_layer, alpha=model.alpha)

            P = torch.sum(torch.nn.Softmax(dim=1)(10*q_batch), axis=0)
            P = P/P.sum()
            Q = torch.ones(args.n_clusters).to(args.device)/args.n_clusters # Uniform distribution

            if args.cluster_balance == "kl":
                cluster_balance_loss = F.kl_div(P.log(), Q, reduction='batchmean')
            else:
                cluster_balance_loss = torch.linalg.norm(torch.sqrt(P) - torch.sqrt(Q))

            # print(p_train.shape, idx_batch)
            km_loss = F.kl_div(q_batch.log(), p_train[idx_batch], reduction='batchmean')

            loss = args.alpha*reconstr_loss
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
            model.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            model.optimizer.step()
            n_batches += 1

        print('Epoch: {:02d} | Batch: {:02d} | Batch KM Loss: {:.3f} | Total Loss: {:.3f} | Classification Loss: {:.3f} |\
        Cluster Balance Loss: {:.3f}'.format(epoch, batch_idx, epoch_km_loss/n_batches, epoch_loss/n_batches,\
        epoch_class_loss/n_batches, epoch_balance_loss/n_batches))
        train_losses.append([np.round(epoch_loss.item(),3), np.round(epoch_class_loss.item(),3)])

        if epoch % args.log_interval == 0:
            x_train = torch.Tensor(np.concatenate(x_train, axis=0)).to(args.device)
            _, _, q_train = model.encoder_forward(x_train, output="decoded")
            p_train = target_distribution(q_train.detach())

        ######################################################
        ######################################################
        ################### Validation #######################
        ######################################################
        ######################################################

        y_true = []
        y_pred = []
        z_val = []
        q_val = []
        x_val = []
        cluster_ids_val = np.array([])
        with torch.no_grad():
            model.eval()
            model.ae.eval() # prep model for evaluation
            for j in range(model.n_clusters):
                model.classifiers[j][0].eval()

            for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in enumerate(batch_iter(X_val, y_val, X_val_len, args.batch_size)):
                model.optimizer.zero_grad()
                x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_features, args.end_t), dtype=torch.float32).to(device)

                # if args.plot == 'True':
                    # plot(model, torch.FloatTensor(x_batch).to(args.device), y_train, args, labels=cluster_indices, epoch=epoch)

                if x_batch.shape[0] < args.n_clusters:
                    continue

                # y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
                y_batch = torch.Tensor(y_batch).type(torch.LongTensor).to(model.device)
                batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
                
                for i in range(len(batch_lens)):
                    batch_lens[i] = min(batch_lens[i], args.end_t)

                masks = length_to_mask(batch_lens).unsqueeze(-1).float()
                x_batch = torch.nan_to_num(x_batch)

                # Evaluate model on Validation dataset
                z_batch, _, q_batch = model.encoder_forward(x_batch, output="decoded")

                cluster_ids = torch.argmax(q_batch, axis=1)
                preds = torch.zeros((len(z_batch), args.n_classes)).to(args.device)

                z_val.append(z_batch.detach().cpu().numpy())
                q_val.append(q_batch.detach().cpu().numpy())
                x_val.append(x_batch.detach().cpu().numpy())

                cluster_ids_val = np.hstack([cluster_ids_val, cluster_ids.detach().cpu().numpy()])
                # Weighted predictions
                if args.attention == False:
                    for j in range(model.n_clusters):
                        cluster_id = np.where(cluster_ids == j)[0]
                        X_cluster = z_batch[cluster_id]
                        cluster_preds_val = model.classifiers[j][0](X_cluster)
                        preds[cluster_id,:] = cluster_preds_val

                else:
                    X_cluster = z_batch
                    cluster_preds = model.classifiers[j][0](X_cluster)
                    # print("cluster_preds ", x_batch.is_cuda)
                    # print("q_batch ", x_batch.is_cuda)
                    for c in range(args.n_classes):
                        preds[:,c] += q_batch[:,j]*cluster_preds[:,c]

                y_pred += list(preds.cpu().detach().cpu().numpy())
                y_true += list(y_batch.cpu().cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        z_val = np.concatenate(z_val, axis=0)
        q_val = np.concatenate(q_val, axis=0)
        x_val = np.concatenate(x_val, axis=0)

        if args.plot == 'True':
            plot(model, torch.FloatTensor(x_val).to(args.device), y_val, args, labels=cluster_ids_val, epoch=epoch)

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

        if args.optimize == 'auc':
            opt = val_auc
        elif args.optimize == 'auprc':
            opt = val_auprc
        else:
            opt = -val_loss

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        es([val_f1, opt], model)

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

    for epoch in range(args.n_epochs):
        for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in enumerate(batch_iter(X_train, y_train, X_train_len, args.batch_size)):
            # To implement
            counter_batch += len(x_batch)
            model.optimizer.zero_grad()
            x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_features, args.end_t), dtype=torch.float32).to(device)
            
            if x_batch.shape[0] < args.n_clusters:
                continue

            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
            batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
            
            for i in range(len(batch_lens)):
                batch_lens[i] = min(batch_lens[i], args.end_t)

            masks = length_to_mask(batch_lens).unsqueeze(-1).float()
            x_batch = torch.nan_to_num(x_batch)
            z_batch, _, q_batch = model.encoder_forward(torch.FloatTensor(x_batch).to(args.device), output="decoded")
            cluster_id_batch = torch.argmax(q_batch, axis=1)

            # Post clustering training
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
                    classifier_labels[j] = np.random.choice(range(args.n_clusters), p = q_batch[j].detach().cpu().numpy())
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
            # for j in range(model.n_clusters):
            #     model.classifiers[j][0].eval()

            # train_preds = torch.zeros((len(z_batch), args.n_classes))
            # train_loss = 0

            # # Weighted predictions
            # z_batch, _, q_batch = model.encoder_forward(torch.FloatTensor(x_batch).to(args.device), output="decoded")

            # cluster_ids_batch = torch.argmax(q_batch, axis=1)
            
            # for j in range(model.n_clusters):
            #     cluster_id = np.where(cluster_ids_batch == j)[0]
            #     X_cluster = z_batch
            #     y_cluster = torch.Tensor(y_batch[cluster_id]).type(torch.LongTensor)

            #     # Ensemble train loss
            #     cluster_preds = model.classifiers[j][0](X_cluster)
            #     for c in range(args.n_classes):
            #         train_preds[:,c] += q_batch[:,j]*cluster_preds[:,c]

            #     X_cluster = z_batch[cluster_id]
            #     cluster_preds = model.classifiers[j][0](X_cluster)
            #     train_loss += torch.sum(q_batch[cluster_id,j]*criterion(cluster_preds, y_cluster))

            # train_loss /= len(z_batch)
            
            # print_msg = (f'\n[{epoch:>{epoch_len}}/{N_EPOCHS:>{epoch_len}}] ' +
            #                  f'train_loss: {train_loss:.3f} ')
            # print(print_msg)
            # print("\n")

        ###########################################
        ## Local training validation performance ##
        ###########################################

        y_true = []
        y_pred = []
        z_val = []
        cluster_ids_val = np.array([])

        for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in enumerate(batch_iter(X_val, y_val, X_val_len, args.batch_size)):
            model.optimizer.zero_grad()
            x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_features, args.end_t), dtype=torch.float32).to(device)
            
            if x_batch.shape[0] < args.n_clusters:
                continue

            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
            batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
            
            for i in range(len(batch_lens)):
                batch_lens[i] = min(batch_lens[i], args.end_t)

            masks = length_to_mask(batch_lens).unsqueeze(-1).float()
            x_batch = torch.nan_to_num(x_batch)

            # Evaluate model on Validation dataset
            # q_batch, z_batch = model(torch.FloatTensor(x_batch).to(args.device), output="latent")
            z_batch, _, q_batch = model.encoder_forward(torch.FloatTensor(x_batch).to(args.device), output="decoded")
            cluster_ids = torch.argmax(q_batch, axis=1)

            # preds = torch.zeros((len(z_batch), args.n_classes))
            preds = model.predict(z=torch.FloatTensor(z_batch).to(args.device), q=q_batch, attention=attention_test)

            # # Weighted predictions
            # for j in range(model.n_clusters):
            #     cluster_id = np.where(cluster_ids == j)[0]
            #     X_cluster = z_batch
            #     cluster_preds = model.classifiers[j][0](X_cluster)
            #     for c in range(args.n_classes):
            #         preds[:,c] += q_batch[:,j]*cluster_preds[:,c]

            y_pred += list(preds.cpu().detach().numpy())
            y_true += list(y_batch.cpu().numpy())
            z_val.append(z_batch.detach().numpy())
            cluster_ids_val = np.hstack([cluster_ids_val, cluster_ids.detach().cpu().numpy()])


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

        if args.optimize == 'auc':
            opt = val_auc
        elif args.optimize == 'auprc':
            opt = val_auprc
        else:
            opt = -val_loss

        es([val_f1, opt], model)
        if es.early_stop == True:
            # train_losses.append(train_loss.item())
            sil_scores.append(silhouette_new(z_val, cluster_ids_val, metric='euclidean'))
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


    for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in enumerate(batch_iter(X_test, y_test, X_test_len, args.batch_size)):
        model.optimizer.zero_grad()
        x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_features, args.end_t), dtype=torch.float32).to(device)
        
        if x_batch.shape[0] < args.n_clusters:
            continue

        y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
        batch_lens = torch.tensor(batch_lens, dtype=torch.float32).to(device).int()
        
        for i in range(len(batch_lens)):
            batch_lens[i] = min(batch_lens[i], args.end_t)

        masks = length_to_mask(batch_lens).unsqueeze(-1).float()
        x_batch = torch.nan_to_num(x_batch)

        # Evaluate model on Validation dataset
        z_batch, _, q_batch = model.encoder_forward(torch.FloatTensor(x_batch).to(args.device), output="decoded")
        cluster_ids = torch.argmax(q_batch, axis=1)
        preds = model.predict(z=torch.FloatTensor(z_batch).to(args.device), q=q_batch, attention=attention_test)

        # # Weighted predictions
        # for j in range(model.n_clusters):
        #     cluster_id = np.where(cluster_ids == j)[0]
        #     X_cluster = z_batch
        #     cluster_preds = model.classifiers[j][0](X_cluster)
        #     for c in range(args.n_classes):
        #         preds[:,c] += q_batch[:,j]*cluster_preds[:,c]

        y_pred += list(preds.cpu().detach().numpy())
        y_true += list(y_batch.cpu().numpy())
        z_test.append(z_batch.detach().cpu().numpy())
        cluster_ids_test = np.hstack([cluster_ids_test, cluster_ids.detach().cpu().numpy()])

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

    test_metrics = performance_metrics(y_batch, preds.detach().cpu().numpy(), args.n_classes)
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
    _ , z_train = model.encoder_forward(torch.FloatTensor(X_train).to(args.device), output='latent')
    WDFD_Single_Cluster_Analysis(X_train, y_train, cluster_ids_train, column_names,\
        scale=scale, X_latents=z_train, dataset=args.dataset, n_results=15)

    HTFD_Single_Cluster_Analysis(scale.inverse_transform(X_train), y_train, cluster_ids_train, column_names,\
        scale=None, n_results=25)