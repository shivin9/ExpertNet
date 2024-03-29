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
from scipy.spatial import distance_matrix

import argparse
import numpy as np
from scipy.cluster.vq import vq, kmeans2
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
from pytorchtools import EarlyStoppingCAC
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import torch.autograd.profiler as profiler

import numbers
from sklearn.metrics import davies_bouldin_score as dbs, adjusted_rand_score as ari
from matplotlib import pyplot as plt
color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

from models import DeepCAC,  target_distribution, source_distribution, CNN_AE, DAE, CIFAR_AE
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
parser.add_argument('--pre_epoch', default= 200, type=int)
parser.add_argument('--pretrain', default= True, type=bool)
parser.add_argument("--load_ae",  default=False, type=bool)
parser.add_argument("--classifier", default="LR")
parser.add_argument("--tol", default=0.01, type=float)
parser.add_argument("--attention", default="True")
parser.add_argument('--ablation', default='None')
parser.add_argument('--cluster_balance', default='hellinger')
parser.add_argument('--ae_type', default= 'dae')
parser.add_argument('--n_channels', default= 1, type=int)
parser.add_argument('--sub_epochs', default= 'False')

# Model parameters
parser.add_argument('--lamda', default= 1, type=float)
parser.add_argument('--beta', default= 0.5, type=float) # KM loss wt
parser.add_argument('--gamma', default= 0.0, type=float) # Classification loss wt
parser.add_argument('--delta', default= 0.0, type=float) # Class equalization wt
parser.add_argument('--eta', default= 0.01, type=float) # Class seploss wt
parser.add_argument('--hidden_dims', default= [64, 32])
parser.add_argument('--n_z', default= 20, type=int)
parser.add_argument('--n_clusters', default= 3, type=int)
parser.add_argument('--clustering', default= 'cac')
parser.add_argument('--n_classes', default= 2, type=int)
parser.add_argument('--optimize', default= 'loss')

# Utility parameters
parser.add_argument('--device', default= 'cpu')
parser.add_argument('--verbose', default= 'False')
parser.add_argument('--plot', default= 'False')
parser.add_argument('--expt', default= 'DeepCAC')
parser.add_argument('--cluster_analysis', default= 'False')
parser.add_argument('--log_interval', default= 10, type=int)
parser.add_argument('--pretrain_path', default= '/Users/shivin/Document/NUS/Research/CAC/CAC_DL/ExpertNet/pretrained_model/DeepCAC')

parser = parser.parse_args()  
args = parameters(parser)

base_suffix = ""
base_suffix += args.dataset
base_suffix += "_" + args.ae_type
base_suffix += "_k_" + str(args.n_clusters)
base_suffix += "_att_" + str(args.attention)
base_suffix += "_dr_" + str(args.data_ratio)
base_suffix += "_target_" + str(args.target)

####################################################################################
####################################################################################
####################################################################################
################################### Initialiation ##################################
####################################################################################
####################################################################################
####################################################################################

f1_scores, auc_scores, auprc_scores, acc_scores, minpse_scores = [], [], [], [], [] #Inattentive test results
sil_scores, wdfd_scores, htfd_scores, w_htfd_scores = [], [], [], []
nmi_scores, ari_scores = [], []

# to track the training loss as the model trains
test_losses, e_test_losses, local_sum_test_losses = [], [], []
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

args.pretrain_path += "/AE_" + base_suffix + ".pth"

for r in range(len(iter_array)):
    scale, column_names, train_data, val_data, test_data = get_train_val_test_loaders(args, r_state=r)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    train_loader = generate_data_loaders(X_train, y_train, args.batch_size)
    val_loader = generate_data_loaders(X_val, y_val, args.batch_size)
    test_loader = generate_data_loaders(X_test, y_test, args.batch_size)

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

    # ae_layers = [128, 64, args.n_z, 64, 128]
    # expert_layers = [args.n_z, 128, 64, 32, 16, args.n_classes]

    ae_layers = [64, args.n_z, 64]
    expert_layers = [args.n_z, args.n_classes]

    if args.ae_type == 'cnn':
        if X_train[0].shape[1] == 28:
            deepcac_ae = CNN_AE(args, fc2_input_dim=128)
        elif X_train[0].shape[1] == 32:
            deepcac_ae = CIFAR_AE(args, fc2_input_dim=128)
    
    else:
        ae_layers.append(args.input_dim)
        ae_layers = [args.input_dim] + ae_layers
        deepcac_ae = DAE(ae_layers)

    model = DeepCAC(
            deepcac_ae,
            expert_layers,
            args=args).to(args.device)

    model.pretrain(train_loader, args.pretrain_path)
    optimizer = Adam(model.parameters(), lr=args.lr_enc)

    # Initiate cluster parameters
    device = args.device
    y = y_train
    x_bar, hidden = model.ae(torch.Tensor(X_train).to(args.device))
    original_cluster_centers, cluster_indices = kmeans2(hidden.data.cpu().numpy(), k=args.n_clusters, minit='++')
    model.cluster_layer.data = torch.tensor(original_cluster_centers).to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    cac_criterion = AdMSoftmaxLoss(args.n_z, args.n_classes, s=30.0, m=0.4) # Default values recommended by [1]

    for i in range(args.n_clusters):
        cluster_idx = np.where(cluster_indices == i)[0]
        for c in range(args.n_classes):
            cluster_idx_c = np.where(y[cluster_idx] == c)[0]
            hidden_c = hidden[cluster_idx][cluster_idx_c]    
            model.class_cluster_layer.data[i,c,:] = torch.mean(hidden_c, axis=0)

    ####################################################################################
    ####################################################################################
    ####################################################################################
    ################################## Clustering Step #################################
    ####################################################################################
    ####################################################################################
    ####################################################################################


    print("Starting Training")
    model.train()
    N_EPOCHS = args.n_epochs
    es = EarlyStoppingCAC(dataset=suffix, patience=5)
    train_losses, e_train_losses = [], []

    for epoch in range(N_EPOCHS):
        alpha = args.alpha
        beta = args.beta
        gamma = args.gamma
        delta = args.delta
        eta = args.eta
        if epoch % args.log_interval == 0:
            model.ae.eval() # prep model for evaluation
            for j in range(model.n_clusters):
                model.classifiers[j][0].eval()

            z_train, x_train_bar = model(torch.FloatTensor(X_train).to(args.device), output='latent')

            # evaluate clustering performance
            cluster_indices, _ = vq(z_train.data.cpu().numpy(), model.cluster_layer.cpu().numpy())

            if args.plot == 'True':
                plot(model, torch.FloatTensor(X_train).to(args.device), y_train, args, labels=cluster_indices, epoch=epoch)

            cluster_indices = torch.Tensor(cluster_indices).type(torch.LongTensor)
            train_sil = silhouette_new(z_train.data.cpu().numpy(), cluster_indices.data.cpu().numpy(), metric='euclidean')

            # Calculate Training Metrics
            nmi, acc, ari = 0, 0, 0
            train_loss, cluster_loss, class_sep_loss = 0, 0, 0
            B = []

            # Training loss
            for j in range(model.n_clusters):
                cluster_idx = np.where(cluster_indices == j)[0]
                if len(cluster_idx) == 0:
                    continue

                X_cluster = z_train[cluster_idx]
                y_cluster = torch.Tensor(y_train[cluster_idx]).type(torch.LongTensor)

                # point wise distanes for every cluster
                cluster_loss += torch.linalg.norm(X_cluster - model.cluster_layer[j].detach())/len(cluster_idx)
                class_sep_loss += cac_criterion(X_cluster, y_cluster)

            reconstr_loss = F.mse_loss(x_train_bar, torch.FloatTensor(X_train).to(args.device), reduction='mean')
            train_loss = alpha*reconstr_loss + beta*cluster_loss + eta*class_sep_loss

            # Evaluate model on Validation dataset
            z_val, x_val_bar = model(torch.FloatTensor(X_val).to(args.device), output='latent')
            cluster_ids_val, _ = vq(z_val.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
            cluster_ids_val = torch.Tensor(cluster_ids_val).type(torch.LongTensor)
            preds = torch.zeros((len(z_val), args.n_classes))

            # Hard Predictions
            for j in range(model.n_clusters):
                cluster_id = np.where(cluster_ids_val == j)[0]
                X_cluster = z_val[cluster_id]
                cluster_preds_val = model.classifiers[j][0](X_cluster)
                preds[cluster_id,:] = cluster_preds_val

            # Classification Matrics
            val_metrics = performance_metrics(y_val, preds.detach().numpy(), args.n_classes)
            val_f1  = val_metrics['f1_score']
            val_auc = val_metrics['auroc']
            val_auprc = val_metrics['auprc']
            val_minpse = val_metrics['minpse']

            # Clustering Metrics
            val_sil = silhouette_new(z_val.data.cpu().numpy(), cluster_ids_val.data.cpu().numpy(), metric='euclidean')
            # sil_scores[r].append(train_sil)
            sil_scores.append(train_sil)

            val_feature_diff, val_WDFD = 0, 0
            # val_feature_diff = calculate_HTFD(X_val, cluster_ids_val)
            # val_WDFD = calculate_WDFD(X_val, cluster_ids_val)
            val_class_loss = torch.mean(criterion(preds, torch.Tensor(y_val).type(torch.LongTensor)))
            reconstr_loss = F.mse_loss(x_val_bar, torch.FloatTensor(X_val).to(args.device), reduction='mean')
            val_loss = reconstr_loss
            val_loss, cluster_loss, class_sep_loss = 0, 0, 0

            for j in range(model.n_clusters):
                cluster_idx = np.where(cluster_ids_val == j)[0]
                if len(cluster_idx) == 0:
                    continue
                X_cluster = z_val[cluster_idx]
                y_cluster = torch.Tensor(y_val[cluster_idx]).type(torch.LongTensor)

                cluster_loss += torch.linalg.norm(X_cluster - model.cluster_layer[j])/len(cluster_idx)
                class_sep_loss += cac_criterion(X_cluster, y_cluster.clone().detach())

            val_loss = alpha*reconstr_loss + beta*cluster_loss + eta*class_sep_loss
            epoch_len = len(str(N_EPOCHS))

            print_msg = (f'\n[{epoch:>{epoch_len}}/{N_EPOCHS:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.3f} ' +
                         f'train_SIL: {train_sil:.3f} '  +
                         f'valid_loss: {val_loss:.3f} '  +
                         f'valid_cluster_loss: {cluster_loss:.3f} '  +
                         f'valid_class_sep_loss: {class_sep_loss:.3f} '  +
                         f'valid_SIL: {val_sil:.3f} ' + 
                         f'valid_Class_Loss: {val_class_loss:.3f} ')

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
                break

        # Normal Training
        epoch_loss = 0
        epoch_sep_loss = 0
        epoch_cluster_loss = 0

        model.ae.train() # prep model for evaluation
        for j in range(model.n_clusters):
            model.classifiers[j][0].train()

        for batch_idx, (x_batch, y_batch, idx) in enumerate(train_loader):
            total_loss = 0
            x_batch = x_batch.to(device)
            idx = idx.to(device)
            # delta_mu   = torch.zeros((args.n_clusters, args.latent_dim)).to(args.device)
            # delta_mu_c = torch.zeros((args.n_clusters, args.n_classes, args.latent_dim)).to(args.device)

            cluster_loss, class_sep_loss, loss = 0, 0, 0

            X_latents, x_bar = model(x_batch, output='latent')
            reconstr_loss = F.mse_loss(x_bar, x_batch, reduction='mean')

            cluster_ids, _ = vq(X_latents.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
            cluster_ids = torch.Tensor(cluster_ids).type(torch.LongTensor)

            for j in range(args.n_clusters):
                cluster_pts_idx = torch.where(cluster_ids == j)[0]
                if len(cluster_pts_idx) > 0:
                    X_cluster = X_latents[cluster_pts_idx]
                    y_cluster = y_batch[cluster_pts_idx]
                    y_clstr = label_binarize(y_batch[cluster_pts_idx], classes=list(range(args.n_classes+1)))[:,:args.n_classes]
                    y_clstr = torch.Tensor(y_clstr).type(torch.LongTensor)

                    # create centroid distance matrix for cluster 'j'
                    class_sep_loss += cac_criterion(X_cluster, y_cluster.clone().detach())
                    cluster_loss += torch.linalg.norm(X_latents[cluster_pts_idx] - model.cluster_layer[j].detach())/len(cluster_pts_idx)

            loss = alpha*reconstr_loss + beta*cluster_loss + eta*class_sep_loss

            epoch_loss += loss
            epoch_cluster_loss += cluster_loss
            epoch_sep_loss += class_sep_loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Update Assignments
            X_latents, x_bar = model(x_batch, output='latent')
            cluster_id, _ = vq(X_latents.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
            cluster_id = torch.Tensor(cluster_id).type(torch.LongTensor)

            # Update all the class centroids
            N_c = 100
            # N_c = len(X_latents)
            counter = torch.ones(args.n_clusters)*N_c
            class_counter = torch.ones(args.n_clusters, args.n_classes)*N_c
            rand_idx = np.random.randint(0, len(X_latents), N_c)

            for pt in rand_idx:
                pt_cluster = cluster_id[pt]
                eta = 1/counter[pt_cluster]
                updated_cluster = ((1 - eta) * model.cluster_layer.data[pt_cluster] +
                               eta * X_latents.data[pt])
                model.cluster_layer.data[pt_cluster] = updated_cluster
                counter[pt_cluster] += 1

                pt_class = y_batch[pt]
                eta = 1/class_counter[pt_cluster][pt_class]
                updated_class_cluster = ((1 - eta) * model.class_cluster_layer.data[pt_cluster][pt_class] +
                               eta * X_latents.data[pt])
                model.class_cluster_layer.data[pt_cluster][pt_class] = updated_class_cluster
                class_counter[pt_cluster][pt_class] += 1

            # for j in range(args.n_clusters):
            #     pts_index = np.where(cluster_id == j)[0]
            #     cluster_pts = X_latents[pts_index]
            #     delta_mu[j,:] = cluster_pts.sum(axis=0)/(1+len(cluster_pts))

            # # Update the positive and negative centroids
            # for j in range(args.n_clusters):
            #     pts_index = np.where(cluster_id == j)[0]
            #     N  = len(pts_index)
            #     model.cluster_layer.data[j:] -= (1/(100+N))*delta_mu[j:]


        print('Epoch: {:02d} | Epoch KM Loss: {:.3f} | Total Loss: {:.3f} | Cluster Sep Loss: {:.3f}'.format(
                    epoch, cluster_loss, epoch_loss, epoch_sep_loss))
        train_losses.append([np.round(epoch_loss.item(),3), np.round(epoch_cluster_loss.item(),3)])

    ####################################################################################
    ####################################################################################
    ####################################################################################
    ################################### Local Training #################################
    ####################################################################################
    ####################################################################################
    ####################################################################################
    print(sil_scores)
    print("\n####################################################################################\n")
    print("Training Local Networks")
    model = es.load_checkpoint(model)

    es = EarlyStoppingCAC(dataset=suffix, patience=5)

    z_train = model(torch.FloatTensor(np.array(X_train)).to(args.device))
    cluster_ids_train, _ = vq(z_train.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
    cluster_ids_train = torch.Tensor(cluster_ids_train).type(torch.LongTensor)
    X_latents_data_loader = list(zip(z_train.to(args.device), y_train))

    train_loader_latents = torch.utils.data.DataLoader(X_latents_data_loader,
        batch_size=1024, shuffle=False)

    sil_scores.append(silhouette_new(z_train.data.cpu().numpy(), cluster_ids_train.data.cpu().numpy(), metric='euclidean'))
    htfd_scores.append(calculate_HTFD(X_train, cluster_ids_train))

    B = []

    # Post clustering training
    for epoch in range(args.n_epochs):
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1 = 0
        acc = 0

        for j in range(model.n_clusters):
            model.classifiers[j][0].train()

        # Full training of local networks
        for batch_idx, (X_latents, y_batch) in enumerate(train_loader_latents):
            classifier_labels = np.zeros(len(X_latents))

            # Choose classifier for a point probabilistically
            classifier_labels, _ = vq(X_latents.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
            classifier_labels = torch.Tensor(classifier_labels).type(torch.LongTensor)

            for k in range(args.n_clusters):
                idx_cluster = np.where(classifier_labels == k)[0]
                X_cluster = X_latents[idx_cluster]
                y_cluster = y_batch[idx_cluster]

                classifier_k, optimizer_k = model.classifiers[k]

                # Do not backprop the error to encoder
                y_pred_cluster = classifier_k(X_cluster.detach())
                cluster_loss = torch.mean(criterion(y_pred_cluster, y_cluster))
                optimizer_k.zero_grad()
                cluster_loss.backward(retain_graph=True)
                optimizer_k.step()
        
        for j in range(model.n_clusters):
            model.classifiers[j][0].eval()

        train_preds = torch.zeros((len(z_train), args.n_classes))
        train_loss = 0

        # Weighted predictions
        z_train = model(torch.FloatTensor(X_train).to(args.device))
        cluster_ids_train, _ = vq(z_train.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
        cluster_ids_train = torch.Tensor(cluster_ids_train).type(torch.LongTensor)

        for j in range(model.n_clusters):
            cluster_id = np.where(cluster_ids_train == j)[0]
            X_cluster = z_train[cluster_id]
            y_cluster = torch.Tensor(y_train[cluster_id]).type(torch.LongTensor)

            # Ensemble train loss
            cluster_preds = model.classifiers[j][0](X_cluster)
            train_preds[cluster_id,:] += cluster_preds
            train_loss += torch.sum(criterion(cluster_preds, y_cluster))
            # B.append(torch.max(torch.linalg.norm(X_cluster, axis=1), axis=0).values)

        train_loss /= len(z_train)
        e_train_loss = torch.mean(criterion(train_preds, torch.Tensor(y_train).type(torch.LongTensor)))

        # Evaluate model on Validation set
        z_val = model(torch.FloatTensor(X_val).to(args.device))
        cluster_ids_val, _ = vq(z_val.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
        cluster_ids_val = torch.Tensor(cluster_ids_val).type(torch.LongTensor)

        preds = torch.zeros((len(z_val), args.n_classes))

        # Hard predictions
        for j in range(model.n_clusters):
            cluster_id = np.where(cluster_ids_val == j)[0]
            X_cluster = z_val[cluster_id]
            cluster_preds = model.classifiers[j][0](X_cluster)
            preds[cluster_id,:] += cluster_preds

        # Classification Matrics
        val_metrics = performance_metrics(y_val, preds.detach().numpy(), args.n_classes)
        val_f1  = val_metrics['f1_score']
        val_auc = val_metrics['auroc']
        val_auprc = val_metrics['auprc']
        val_minpse = val_metrics['minpse']

        val_sil = silhouette_new(z_val.data.cpu().numpy(), cluster_ids_val.data.cpu().numpy(), metric='euclidean')
        val_loss = torch.mean(criterion(preds, torch.Tensor(y_val).type(torch.LongTensor)))
        epoch_len = len(str(N_EPOCHS))
        
        print_msg = (f'\n[{epoch:>{epoch_len}}/{N_EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.3f} ' +
                     f'valid_loss: {val_loss:.3f} '  +
                     f'valid_F1: {val_f1:.3f} '  +
                     f'valid_AUC: {val_auc:.3f} ' +
                     f'valid_AUPRC: {val_auprc:.3f} ' +
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
    z_test = model(torch.FloatTensor(X_test).to(args.device))
    cluster_ids, _ = vq(z_test.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
    cluster_ids = torch.Tensor(cluster_ids).type(torch.LongTensor)

    test_loss = 0
    local_class_loss = 0
    test_preds = torch.zeros((len(z_test), args.n_classes))

    # Hard local predictions
    for j in range(model.n_clusters):
        cluster_id = np.where(cluster_ids == j)[0]
        X_cluster = z_test[cluster_id]
        y_cluster = torch.Tensor(y_test[cluster_id]).type(torch.LongTensor)
        cluster_test_preds = model.classifiers[j][0](X_cluster)
        test_preds[cluster_id,:] = cluster_test_preds
        local_class_loss += torch.sum(criterion(cluster_test_preds, y_cluster))
    
    test_metrics = performance_metrics(y_test, test_preds.detach().numpy(), args.n_classes)
    test_f1  = test_metrics['f1_score']
    test_auc = test_metrics['auroc']
    test_auprc = test_metrics['auprc']
    test_minpse = test_metrics['minpse']
    test_acc = test_metrics['acc']

    test_loss = torch.mean(criterion(test_preds, torch.Tensor(y_test).type(torch.LongTensor)))
    local_class_loss /= len(X_test)

    test_losses.append(test_loss.item())
    local_sum_test_losses.append(local_class_loss.item())

    print("Run #{}".format(r))
    print('Loss Metrics - Test Loss {:.3f}, Local Sum Test Loss {:.3f}'.format(test_loss, local_class_loss))

    print('Clustering Metrics     - Acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi),\
          ', ari {:.4f}'.format(ari))

    print('Classification Metrics - Test F1 {:.3f}, Test AUC {:.3f}, Test AUPRC {:.3f},\
        Test ACC {:.3f}'.format(test_f1, test_auc, test_auprc, test_acc))

    print("\n")

    f1_scores.append(test_f1)
    auc_scores.append(test_auc)
    acc_scores.append(test_acc)
    auprc_scores.append(test_auprc)
    minpse_scores.append(test_minpse)
    nmi_scores.append(nmi_score(cluster_ids.data.cpu().numpy(), y_test))
    ari_scores.append(ari_score(cluster_ids.data.cpu().numpy(), y_test))
    # sil_scores.append(silhouette_new(z_test.data.cpu().numpy(), cluster_ids.data.cpu().numpy(), metric='euclidean'))

    ###################################################################################
    ###################################################################################
    ###################################################################################
    ################################## Feature Imp. ###################################
    ###################################################################################
    ###################################################################################
    ###################################################################################

    # regs = [GradientBoostingClassifier(random_state=0) for _ in range(args.n_clusters)]
    # for r in regs:
    #     if args.n_classes > 2:
    #         r.predict_proba = lambda X: r.decision_function(X)
    #     else:
    #         r.predict_proba = lambda X: np.array([r.decision_function(X), r.decision_function(X)]).transpose()

    # z_train = model(torch.FloatTensor(X_train).to(args.device))
    # cluster_ids, _ = vq(z_train.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
    # cluster_ids = torch.Tensor(cluster_ids).type(torch.LongTensor)

    # train_preds = torch.zeros((len(z_train), args.n_classes))
    # feature_importances = np.zeros((args.n_clusters, args.input_dim))

    # # Weighted predictions... should be without attention only
    # for j in range(model.n_clusters):
    #     cluster_id = torch.where(cluster_ids == j)[0]
    #     X_cluster = z_train[cluster_id]
    #     cluster_preds = model.classifiers[j][0](X_cluster)
    #     train_preds[cluster_id,:] = cluster_preds

    #     # train student on teacher's predictions
    #     # y_cluster = np.argmax(train_preds[cluster_id].detach().numpy(), axis=1)
        
    #     # train student on real labels
    #     y_cluster = y_train[cluster_id]

    #     # Train the local regressors on the data embeddings
    #     # Some test data might not belong to any cluster
    #     if len(cluster_id) > 0:
    #         regs[j].fit(X_train[cluster_id], y_cluster)
    #         best_features = np.argsort(regs[j].feature_importances_)[::-1][:10]
    #         feature_importances[j,:] = regs[j].feature_importances_
    #         print("Cluster # ", j, "sized: ", len(cluster_id), "label distr: ", np.bincount(y_cluster)/len(y_cluster))
    #         print(list(zip(column_names[best_features], np.round(regs[j].feature_importances_[best_features], 3))))
    #         print("=========================\n")

    # # Testing performance of downstream classifier on cluster embeddings
    # z_test = model(torch.FloatTensor(X_test).to(args.device))
    # cluster_ids_test, _ = vq(z_test.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
    # cluster_ids_test = torch.Tensor(cluster_ids_test).type(torch.LongTensor)

    # test_preds = torch.zeros((len(z_test), args.n_classes))
    # y_pred = np.zeros((len(z_test), args.n_classes))

    # for j in range(model.n_clusters):
    #     cluster_id = torch.where(cluster_ids_test == j)[0]
    #     X_cluster = X_test[cluster_id]

    #     # Some test data might not belong to any cluster
    #     if len(cluster_id) > 0:
    #         y_pred[cluster_id] = regs[j].predict_proba(X_cluster)

    # test_metrics = performance_metrics(y_test, y_pred, args.n_classes)
    # test_f1  = test_metrics['f1_score']
    # test_auc = test_metrics['auroc']
    # test_auprc = test_metrics['auprc']
    # test_minpse = test_metrics['minpse']
    # test_acc = test_metrics['acc']

    # print('Student Network Classification Metrics - Test F1 {:.3f}, Test AUC {:.3f}, Test AUPRC {:.3f},\
    #     Test ACC {:.3f}'.format(test_f1, test_auc, test_auprc, test_acc))

    # print("\n")

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
    #     w_htfd_scores.append(0)
    # else:
    #     print("Average Feature Difference: ", feature_diff/cntr)
    #     w_htfd_scores.append(feature_diff/cntr)


enablePrint()

print("Test F1: ", f1_scores)
print("Test AUC: ", auc_scores)
print("Test AUPRC: ", auprc_scores)
print("Test MINPSE: ", minpse_scores)
print("Train SIL: ", sil_scores)

# print("\n")
# print("Experiment ", iteration_name)
# print(iter_array)
# print("Test F1: ", f1_scores)
# print("Test AUC: ", auc_scores)
# print("Test AUPRC: ", auprc_scores)

# print("Sil scores: ", sil_scores)
# print("HTFD: ", htfd_scores)

# print("Train Cluster Counts: ", np.bincount(cluster_ids_train))

# print("Train Loss: ", train_losses)

# print("Test Loss: ", test_losses)
# print("Local Test Loss: ", local_sum_test_losses)

# print("Model Complexity: ", model_complexity)

# print('Dataset\tk')
# print("{}\t{}\n".format(args.dataset, args.n_clusters))

for key in ['alpha', 'beta', 'gamma', 'eta', 'data_ratio', 'n_clusters', 'attention']:
    print(key, args.__dict__[key])

print("[Avg]\tDataset\tk\tF1\tAUC\tAUPRC\tMINPSE\tACC\tTr-SIL\tTr-HTFD\tTe-NMI\tTe-ARI")

print("\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format\
    (args.dataset, args.n_clusters, np.avg(f1_scores), np.avg(auc_scores),\
    np.avg(auprc_scores), np.avg(minpse_scores), np.avg(acc_scores),\
    np.avg(np.array(sil_scores)), np.avg(np.array(htfd_scores)), np.avg(nmi_scores), np.avg(ari_scores)))

print("[Std]\tF1\tAUC\tAUPRC\tMINPSE\tACC\tTe-SIL\tTr-HTFD\tTe-NMI\tTe-ARI")

print("\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format\
    (np.std(f1_scores), np.std(auc_scores),np.std(auprc_scores),\
    np.std(minpse_scores), np.std(acc_scores), np.std(np.array(sil_scores)),\
    np.std(np.array(htfd_scores)), np.std(nmi_scores), np.std(ari_scores)))

# print("Train Loss\tTest Loss")

# print("{:.3f}\t{:.3f}\n".format\
#     (np.avg(train_losses), np.avg(test_losses)))

if args.cluster_analysis == "True":
    # WDFD_Cluster_Analysis(torch.Tensor(X_train), cluster_ids_train, column_names)
    # HTFD_Cluster_Analysis(torch.Tensor(X_train), cluster_ids_train, column_names)
    HTFD_Cluster_Analysis(scale.inverse_transform(X_train), y_train, cluster_ids_train, column_names)
    HTFD_Single_Cluster_Analysis(scale.inverse_transform(X_train), y_train, cluster_ids_train, column_names)