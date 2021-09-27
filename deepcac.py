from __future__ import print_function, division
import copy
import torch
import argparse
import numpy as np
import umap
import os
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score,\
f1_score, roc_auc_score, roc_curve, accuracy_score, matthews_corrcoef as mcc
from torch.utils.data import Subset
import pandas as pd
from scipy.stats import ttest_ind
from scipy.spatial import distance_matrix

import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score, silhouette_score
from sklearn.ensemble import GradientBoostingRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.nn import Linear
from pytorchtools import EarlyStoppingCAC

import numbers
from sklearn.metrics import davies_bouldin_score as dbs, adjusted_rand_score as ari
from matplotlib import pyplot as plt
color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

from models import MultiHeadIDEC,  target_distribution, source_distribution
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
parser.add_argument('--pre_epoch', default= 40, type=int)
parser.add_argument('--pretrain', default= True, type=bool)
parser.add_argument("--load_ae",  default=False, type=bool)
parser.add_argument("--classifier", default="LR")
parser.add_argument("--tol", default=0.01, type=float)
parser.add_argument("--attention", default="True")

# Model parameters
parser.add_argument('--lamda', default= 1, type=float)
parser.add_argument('--beta', default= 0.5, type=float) # KM loss wt
parser.add_argument('--gamma', default= 1.0, type=float) # Classification loss wt
parser.add_argument('--delta', default= 0.01, type=float) # Class seploss wt
parser.add_argument('--hidden_dims', default= [64, 32])
parser.add_argument('--n_z', default= 20, type=int)
parser.add_argument('--n_clusters', default= 3, type=int)
parser.add_argument('--clustering', default= 'cac')
parser.add_argument('--n_classes', default= 2, type=int)

# Utility parameters
parser.add_argument('--device', default= 'cpu')
parser.add_argument('--ablation', default='None')
parser.add_argument('--log_interval', default= 10, type=int)
parser.add_argument('--pretrain_path', default= '/Users/shivin/Document/NUS/Research/CAC/CAC_DL/DeepCAC/pretrained_model')

parser = parser.parse_args()  

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
        self.pre_epoch = parser.pre_epoch
        self.pretrain = parser.pretrain
        self.load_ae = parser.load_ae
        self.classifier = parser.classifier
        self.tol = parser.tol
        self.attention = parser.attention == "True"

        # Model parameters
        self.lamda = parser.lamda
        self.beta = parser.beta
        self.gamma = parser.gamma
        self.delta = parser.delta
        self.hidden_dims = parser.hidden_dims
        self.latent_dim = self.n_z = parser.n_z
        self.n_clusters = parser.n_clusters
        self.clustering = parser.clustering
        self.n_classes = parser.n_classes

        # Utility parameters
        self.device = parser.device
        self.ablation = parser.ablation
        self.log_interval = parser.log_interval
        self.pretrain_path = parser.pretrain_path + "/" + self.dataset + ".pth"

args = parameters(parser)
base_suffix = ""

for key in args.__dict__:
    print(key, args.__dict__[key])

base_suffix += args.dataset + "_"
base_suffix += str(args.n_clusters) + "_"
base_suffix += str(args.attention)

column_names, train_data, val_data, test_data = get_train_val_test_loaders(args)
X_train, y_train, train_loader = train_data
X_val, y_val, val_loader = val_data
X_test, y_test, test_loader = test_data

####################################################################################
####################################################################################
####################################################################################
################################### Initialiation ##################################
####################################################################################
####################################################################################
####################################################################################

f1_scores, auc_scores = [], []


if args.ablation == "beta":
    iter_array = betas
    iteration_name = "Beta"

elif args.ablation == "gamma":
    iter_array = gammas
    iteration_name = "Gamma"

elif args.ablation == "delta":
    iter_array = deltas
    iteration_name = "Delta"

else:
    iter_array = range(5)
    iteration_name = "Run"

for r in range(len(iter_array)):
    print(iteration_name, ":", iter_array[r])
    # blockPrint()

    if args.ablation == "beta":
        args.beta = iter_array[r]

    elif args.ablation == "gamma":
        args.gamma = iter_array[r]

    elif args.ablation == "delta":
        args.delta = iter_array[r]

    suffix = base_suffix + "_" + iteration_name + "_" + str(iter_array[r])
    model = MultiHeadIDEC(
            n_enc_1=128,
            n_enc_2=64,
            n_enc_3=32,
            n_dec_1=32,
            n_dec_2=64,
            n_dec_3=128,
            args=args).to(args.device)

    model.pretrain(train_loader, args.pretrain_path)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # cluster parameter initiate
    device = args.device
    y = y_train
    x_bar, hidden = model.ae(torch.Tensor(X_train).to(args.device))

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    cluster_indices = kmeans.fit_predict(hidden.data.cpu().numpy())
    original_cluster_centers = kmeans.cluster_centers_
    model.cluster_layer.data = torch.tensor(original_cluster_centers).to(device)

    ## Initialization ##
    for i in range(args.n_clusters):
        cluster_idx = np.where(cluster_indices == i)[0]
        cluster_idx_p = np.where(y[cluster_idx] == 1)[0]
        cluster_idx_n = np.where(y[cluster_idx] == 0)[0]
        hidden_p = hidden[cluster_idx][cluster_idx_p]
        hidden_n = hidden[cluster_idx][cluster_idx_n]
        
        model.p_cluster_layer.data[i,:] = torch.mean(hidden_p, axis=0)
        model.n_cluster_layer.data[i,:] = torch.mean(hidden_n, axis=0)

    criterion = nn.CrossEntropyLoss(reduction='mean')


    ####################################################################################
    ####################################################################################
    ####################################################################################
    ################################## Clustering Step #################################
    ####################################################################################
    ####################################################################################
    ####################################################################################


    print("Starting Training")
    model.train()
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    N_EPOCHS = args.n_epochs
    es = EarlyStoppingCAC(dataset=suffix)

    for epoch in range(N_EPOCHS):
        if epoch % args.log_interval == 0:
            blockPrint()
            model.ae.eval() # prep model for evaluation
            for j in range(model.n_clusters):
                model.classifiers[j][0].eval()

            z_train, _, q_train = model(torch.Tensor(X_train).to(args.device), output="decoded")
            q_train, q_train_p, q_train_n = q_train
            # update target distribution p
            q_train = q_train.data

            # evaluate clustering performance
            cluster_indices = q_train.cpu().numpy().argmax(1)

            # Calculate Training Metrics
            nmi, acc, ari = 0, 0, 0
            train_loss = 0
            for j in range(args.n_clusters):
                # kmeans = KMeans(n_clusters=args.n_classes, n_init=20)
                cluster_idx = np.where(cluster_indices == j)[0]
                # y_pred_idx = kmeans.fit_predict(z_train.data.cpu().numpy()[cluster_idx])
                # nmi_k = nmi_score(y_pred_idx, y[cluster_idx])
                # nmi += nmi_k * len(cluster_idx)/len(X_train)
                # acc += cluster_acc(y_pred_idx, y[cluster_idx]) * len(cluster_idx)/len(X_train)
                # ari += ari_score(y_pred_idx, y[cluster_idx]) * len(cluster_idx)/len(X_train)

                X_cluster = z_train[cluster_idx]
                y_cluster = torch.Tensor(y_train[cluster_idx]).type(torch.LongTensor).to(model.device)

                classifier_k, optimizer_k = model.classifiers[j]
                y_pred_cluster = classifier_k(X_cluster)
                cluster_los = criterion(y_pred_cluster, y_cluster)
                train_loss += cluster_los
            
            # Evaluate model on Test dataset
            qs, z_val = model(torch.FloatTensor(X_val).to(args.device), output="latent")
            q_val = qs[0]
            cluster_ids = torch.argmax(q_val, axis=1)
            preds = torch.zeros((len(z_val), 2))

            # Weighted predictions
            if args.attention == False:
                for j in range(model.n_clusters):
                    cluster_id = np.where(cluster_ids == j)[0]
                    X_cluster = z_val[cluster_id]
                    cluster_preds_val = model.classifiers[j][0](X_cluster)
                    preds[cluster_id,:] = cluster_preds_val

            else:
                for j in range(model.n_clusters):
                    cluster_id = np.where(cluster_ids == j)[0]
                    X_cluster = z_val
                    cluster_preds = model.classifiers[j][0](X_cluster)
                    preds[:,0] += q_val[:,j]*cluster_preds[:,0]
                    preds[:,1] += q_val[:,j]*cluster_preds[:,1]

            feature_diff = 0
            cntr = 0
            for i in range(args.n_clusters):
                for j in range(args.n_clusters):
                    if i > j:
                        ci = torch.where(cluster_ids == i)[0]
                        cj = torch.where(cluster_ids == j)[0]
                        Xi = X_val[ci]
                        Xj = X_val[cj]
                        # feature_diff += sum(ttest_ind(Xi, Xj, axis=0)[1] < 0.05)/args.input_dim
                        # print("Cluster [{}, {}] p-value: ".format(i,j), feature_diff)
                        cntr += 1

            print("qval", torch.sum(q_val, axis=0))
            print("qval max", np.bincount(cluster_ids))
            # print("KL div", torch.kl_div(torch.sum(q_val, axis=0),\
            #                         torch.ones(args.n_clusters)/args.n_clusters))
            # val_sil = silhouette_score(z_val.data.cpu().numpy(), cluster_ids.data.cpu().numpy(), metric='euclidean')
            val_sil = 0
            val_f1  = f1_score(y_val, np.argmax(preds.detach().numpy(), axis=1))
            val_auc = roc_auc_score(y_val, preds[:,1].detach().numpy())
            val_feature_diff = feature_diff/cntr

            loss = criterion(preds, torch.Tensor(y_val).type(torch.LongTensor))
            # record validation loss
            valid_losses.append(loss.item())

            # calculate average loss over an epoch
            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)
            
            epoch_len = len(str(N_EPOCHS))
            
            print_msg = (f'\n[{epoch:>{epoch_len}}/{N_EPOCHS:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.3f} ' +
                         f'valid_loss: {valid_loss:.3f} '  +
                         f'valid_F1: {val_f1:.3f} '  +
                         f'valid_AUC: {val_auc:.3f} ' + 
                         f'valid_Feature_p: {val_feature_diff:.3f} ' + 
                         f'valid_Silhouette: {val_sil:.3f}')
            
            print(print_msg)
            
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            es([val_f1, val_auc], model)
            if es.early_stop == True:
                break

        # Normal Training
        epoch_loss = 0
        epoch_balance_loss = 0
        epoch_class_loss = 0

        model.ae.train() # prep model for evaluation
        for j in range(model.n_clusters):
            model.classifiers[j][0].train()

        for batch_idx, (x_batch, y_batch, idx) in enumerate(train_loader):
            # torch.autograd.set_detect_anomaly(True)
            x_batch = x_batch.to(device)
            idx = idx.to(device)

            X_latents, x_bar, q_train = model(x_batch)
            q_train, q_train_p, q_train_n = q_train
            reconstr_loss = F.mse_loss(x_bar, x_batch)

            classifier_labels = np.zeros(len(idx))
            sub_epochs = min(1, 10 - int(epoch/5))
            # sub_epochs = 10
            if args.attention == False:
                classifier_labels = np.argmax(q_train.detach().cpu().numpy(), axis=1)

            for _ in range(sub_epochs):
                # Choose classifier for a point probabilistically
                if args.attention == True:
                    for j in range(len(idx)):
                        classifier_labels[j] = np.random.choice(range(args.n_clusters), p = q_train[j].detach().numpy())

                for k in range(args.n_clusters):
                    idx_cluster = np.where(classifier_labels == k)[0]
                    X_cluster = X_latents[idx_cluster]
                    y_cluster = y_batch[idx_cluster]

                    classifier_k, optimizer_k = model.classifiers[k]
                    # Do not backprop the error to encoder
                    y_pred_cluster = classifier_k(X_cluster.detach())
                    cluster_loss = criterion(y_pred_cluster, y_cluster)
                    optimizer_k.zero_grad()
                    cluster_loss.backward(retain_graph=True)
                    optimizer_k.step()

            class_loss = torch.tensor(0.).to(args.device)
            for k in range(args.n_clusters):
                idx_cluster = np.where(classifier_labels == k)[0]
                X_cluster = X_latents[idx_cluster]
                y_cluster = y_batch[idx_cluster]

                classifier_k, optimizer_k = model.classifiers[k]
                y_pred_cluster = classifier_k(X_cluster)
                cluster_los = criterion(y_pred_cluster, y_cluster)
                class_loss += cluster_los

            delta_mu   = torch.zeros((args.n_clusters, args.latent_dim)).to(args.device)
            cluster_id = torch.argmax(q_train, 1)
            
            km_loss             = 0
            cluster_balance_loss = 0

            for j in range(args.n_clusters):
                pts_index = np.where(cluster_id == j)[0]
                cluster_pts = X_latents[pts_index]
                delta_mu[j,:]   = cluster_pts.sum(axis=0)/(1+len(cluster_pts))
                km_loss += torch.linalg.vector_norm(X_latents[pts_index] - model.cluster_layer[j])/(1+len(cluster_pts))

            q_train = source_distribution(X_latents, model.cluster_layer, alpha=model.alpha)
            P = torch.sum(torch.nn.Softmax(dim=1)(10*q_train), axis=0)
            Q = torch.ones(args.n_clusters)/args.n_clusters # Uniform distribution

            # cluster_balance_loss = F.kl_div(P.log(), Q, reduction='batchmean')
            cluster_balance_loss = torch.linalg.vector_norm(torch.sqrt(P/sum(P)) - torch.sqrt(Q))

            loss = reconstr_loss
            if args.beta != 0:
                loss += args.beta*km_loss
            if args.gamma != 0:
                loss += args.gamma*class_loss
            if args.delta != 0:
                loss += args.delta*cluster_balance_loss

            epoch_loss += loss
            epoch_balance_loss += cluster_balance_loss
            epoch_class_loss += class_loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # Update the positive and negative centroids
            for j in range(args.n_clusters):
                pts_index = np.where(cluster_id == j)[0]
                N  = len(pts_index)
                model.cluster_layer.data[j:]   -= (1/(100+N))*delta_mu[j:]

        print('Epoch: {:02d} | Loss: {:.3f} | Classification Loss: {:.3f} | Cluster Balance Loss: {:.3f}'.format(
                    epoch, epoch_loss, epoch_class_loss, epoch_balance_loss))

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

    es = EarlyStoppingCAC(dataset=suffix)

    qs, z_train = model(torch.FloatTensor(np.array(X_train)).to(args.device), output="latent")
    q_train = qs[0]
    cluster_id_train = torch.argmax(q_train, axis=1)

    # X_latents_data_loader = list(zip(z_train, cluster_id_train, y_train))
    X_latents_data_loader = list(zip(z_train.to(args.device),q_train, y_train))

    train_loader_latents = torch.utils.data.DataLoader(X_latents_data_loader,
        batch_size=1024, shuffle=False)

    # plot(model, torch.FloatTensor(np.array(X_train)).to(args.device), y_train,\
    #      torch.FloatTensor(np.array(X_test)).to(args.device), y_test)

    # Post clustering training
    for e in range(N_EPOCHS):
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1 = 0
        acc = 0

        # model.ae.train() # prep model for evaluation
        for j in range(model.n_clusters):
            model.classifiers[j][0].train()

        # Full training of local networks
        for batch_idx, (X_latents, q_batch, y_batch) in enumerate(train_loader_latents):
            # torch.autograd.set_detect_anomaly(True)

            classifier_labels = np.zeros(len(X_latents))
            # Choose classifier for a point probabilistically
            if args.attention == True:
                for j in range(len(X_latents)):
                    classifier_labels[j] = np.random.choice(range(args.n_clusters), p = q_batch[j].detach().numpy())
            else:
                classifier_labels = torch.argmax(q_batch, axis=1).data.cpu().numpy()
            for k in range(args.n_clusters):
                idx_cluster = np.where(classifier_labels == k)[0]
                X_cluster = X_latents[idx_cluster]
                y_cluster = y_batch[idx_cluster]

                classifier_k, optimizer_k = model.classifiers[k]
                # Do not backprop the error to encoder
                y_pred_cluster = classifier_k(X_cluster.detach())
                cluster_loss = criterion(y_pred_cluster, y_cluster)
                optimizer_k.zero_grad()
                cluster_loss.backward(retain_graph=True)
                optimizer_k.step()
        
        # model.ae.eval() # prep model for evaluation
        for j in range(model.n_clusters):
            model.classifiers[j][0].eval()

        # Evaluate model on Validation set
        qs, z_val = model(torch.FloatTensor(X_val).to(args.device), output="latent")
        q_val = qs[0]
        cluster_ids = torch.argmax(q_val, axis=1)
        preds = torch.zeros((len(z_val), 2))

        # Weighted predictions
        for j in range(model.n_clusters):
            cluster_id = np.where(cluster_ids == j)[0]
            X_cluster = z_val
            cluster_preds = model.classifiers[j][0](X_cluster)
            preds[:,0] += q_val[:,j]*cluster_preds[:,0]
            preds[:,1] += q_val[:,j]*cluster_preds[:,1]

        val_f1  = f1_score(y_val, np.argmax(preds.detach().numpy(), axis=1))
        val_auc = roc_auc_score(y_val, preds[:,1].detach().numpy())

        loss = criterion(preds, torch.Tensor(y_val).type(torch.LongTensor))
        # record validation loss
        valid_losses.append(loss.item())

        # calculate average loss over an epoch
        valid_loss = np.average(valid_losses)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(N_EPOCHS))
        
        print_msg = (f'\n[{epoch:>{epoch_len}}/{N_EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.3f} ' +
                     f'valid_loss: {valid_loss:.3f} '  +
                     f'valid_F1: {val_f1:.3f} '  +
                     f'valid_AUC: {val_auc:.3f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        es([val_f1, val_auc], model)
        if es.early_stop == True:
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

    # # Evaluate model on Test dataset
    qs, z_test = model(torch.FloatTensor(X_test).to(args.device), output="latent")
    q_test = qs[0]
    # cluster_ids = torch.argmax(q_test, axis=1)
    cluster_ids = np.argmax(distance_matrix(z_test.data.cpu().numpy(), model.cluster_layer.data.cpu().numpy()), axis=1)
    preds_e = torch.zeros((len(z_test), 2))

    # Weighted predictions
    for j in range(model.n_clusters):
        cluster_id = np.where(cluster_ids == j)[0]
        # X_cluster = z_test[cluster_id]
        X_cluster = z_test
        cluster_preds = model.classifiers[j][0](X_cluster)
        # print(q_test, cluster_preds[:,0])
        preds_e[:,0] += q_test[:,j]*cluster_preds[:,0]
        preds_e[:,1] += q_test[:,j]*cluster_preds[:,1]

    e_test_f1 = f1_score(y_test, np.argmax(preds_e.detach().numpy(), axis=1))
    e_test_auc = roc_auc_score(y_test, preds_e[:,1].detach().numpy())
    e_test_acc = accuracy_score(y_test, np.argmax(preds_e.detach().numpy(), axis=1))
    e_test_loss = criterion(preds_e, torch.Tensor(y_test).type(torch.LongTensor))

    preds = torch.zeros((len(z_test), 2))

    # Hard local predictions
    for j in range(model.n_clusters):
        cluster_id = np.where(cluster_ids == j)[0]
        X_cluster = z_test[cluster_id]
        cluster_preds = model.classifiers[j][0](X_cluster)
        preds[cluster_id,:] = cluster_preds

    test_f1 = f1_score(y_test, np.argmax(preds.detach().numpy(), axis=1))
    test_auc = roc_auc_score(y_test, preds[:,1].detach().numpy())
    test_acc = accuracy_score(y_test, np.argmax(preds.detach().numpy(), axis=1))
    test_loss = criterion(preds, torch.Tensor(y_test).type(torch.LongTensor))

    print('Run #{}, Acc {:.4f}'.format(r, acc),
          ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),\
          ', Test Loss {:.3f}, E-Test Loss {:.3f}'.format(test_loss, e_test_loss))

    enablePrint()
    print('Run #{}, Test F1 {:.3f}, Test AUC {:.3f}, Test ACC {:.3f}'.format(r, test_f1, test_auc, test_acc),\
        ', E-Test F1 {:.3f}, E-Test AUC {:.3f}, E-Test ACC {:.3f}'.format(e_test_f1, e_test_auc, e_test_acc))

    f1_scores.append(e_test_f1)
    auc_scores.append(e_test_auc)

    ####################################################################################
    ####################################################################################
    ####################################################################################
    ################################### Feature Imp. ###################################
    ####################################################################################
    ####################################################################################
    ####################################################################################


    # regs = [GradientBoostingRegressor(random_state=0) for _ in range(args.n_clusters)]
    # qs, z_train = model(torch.FloatTensor(X_train).to(args.device), output="latent")
    # q_train = qs[0]
    # cluster_ids = torch.argmax(q_train, axis=1)
    # preds_e = torch.zeros((len(z_train), 2))
    # feature_importances = np.zeros((args.n_clusters, args.input_dim))

    # # Weighted predictions
    # for j in range(model.n_clusters):
    #     X_cluster = z_train
    #     cluster_preds = model.classifiers[j][0](X_cluster)
    #     # print(q_test, cluster_preds[:,0])
    #     preds_e[:,0] += q_train[:,j]*cluster_preds[:,0]
    #     preds_e[:,1] += q_train[:,j]*cluster_preds[:,1]

    # for j in range(model.n_clusters):
    #     cluster_id = torch.where(cluster_ids == j)[0]
    #     X_cluster = X_train[cluster_id]
    #     if args.attention == True:
    #         y_cluster = preds_e[cluster_id][:,1]
    #     else:
    #         y_cluster = preds[cluster_id][:,1]

    #     # Some test data might not belong to any cluster
    #     if len(cluster_id) > 0:
    #         regs[j].fit(X_cluster, y_cluster.detach().cpu().numpy())
    #         best_features = np.argsort(regs[j].feature_importances_)[::-1][:10]
    #         feature_importances[j,:] = regs[j].feature_importances_
    #         print("Cluster # ", j, "sized: ", len(cluster_id))
    #         print(list(zip(column_names[best_features], np.round(regs[j].feature_importances_[best_features], 3))))
    #         print("=========================\n")

    # feature_diff = 0
    # cntr = 0
    # for i in range(args.n_clusters):
    #     for j in range(args.n_clusters):
    #         if i > j:
    #             ci = torch.where(cluster_ids == i)[0]
    #             cj = torch.where(cluster_ids == j)[0]
    #             Xi = X_train[ci]
    #             Xj = X_train[cj]
    #             feature_diff += sum(feature_importances[i]*feature_importances[j]*(ttest_ind(Xi, Xj, axis=0)[1] < 0.05))/args.input_dim
    #             print("Cluster [{}, {}] p-value: ".format(i,j), feature_diff)
    #             cntr += 1

    # print("Average Feature Difference: ", feature_diff/cntr)

print("Avg. Test F1 = {:.3f}, AUC = {:.3f}".format(np.average(f1_scores), np.average(auc_scores)))