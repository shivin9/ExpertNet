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

import numbers
from sklearn.metrics import davies_bouldin_score as dbs, adjusted_rand_score as ari
from matplotlib import pyplot as plt
color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

from models import DeepCAC,  target_distribution, source_distribution
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
parser.add_argument('--gamma', default= 0.0, type=float) # Classification loss wt
parser.add_argument('--delta', default= 0.0, type=float) # Class equalization wt
parser.add_argument('--eta', default= 0.01, type=float) # Class seploss wt
parser.add_argument('--hidden_dims', default= [64, 32])
parser.add_argument('--n_z', default= 20, type=int)
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

softmin = torch.nn.Softmin(dim=0)
U = torch.eye(args.n_classes, args.n_classes)*1000

####################################################################################
####################################################################################
####################################################################################
################################### Initialiation ##################################
####################################################################################
####################################################################################
####################################################################################

f1_scores, auc_scores, auprc_scores, acc_scores, minpse_scores = [], [], [], [], [] #Inattentive test results
sil_scores, wdfd_scores, HTFD_scores, w_HTFD_scores = [], [], [], []

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
    # ae_layers = [128, 64, 32, args.n_z, 32, 64, 128]
    ae_layers = [64, args.n_z, 64]
    expert_layers = [args.n_z, 30, args.n_classes]

    model = DeepCAC(
            ae_layers,
            expert_layers,
            args=args).to(args.device)

    model.pretrain(train_loader, args.pretrain_path)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Initiate cluster parameters
    device = args.device
    y = y_train
    x_bar, hidden = model.ae(torch.Tensor(X_train).to(args.device))
    original_cluster_centers, cluster_indices = kmeans2(hidden.data.cpu().numpy(), k=args.n_clusters, minit='++')
    model.cluster_layer.data = torch.tensor(original_cluster_centers).to(device)
    # kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(X_train.data)
    # model.cluster_layer.data = torch.tensor(kmeans.labels_).to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    cac_criterion = AdMSoftmaxLoss(args.n_z, args.n_classes, s=30.0, m=0.4) # Default values recommended by [1]
    # cac_criterion = nn.MultiMarginLoss(reduction='mean')
    # print("Original cluster counts: ", np.bincount(cluster_indices))
    # plot(model, torch.FloatTensor(np.array(X_train)).to(args.device), y_train, labels=cluster_indices)

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
    es = EarlyStoppingCAC(dataset=suffix)
    train_losses, e_train_losses = [], []

    for epoch in range(N_EPOCHS):
        beta = args.beta
        gamma = args.gamma
        delta = args.delta
        eta = args.eta
        if epoch % args.log_interval == 0:
            # plot(model, torch.FloatTensor(X_val).to(args.device), y_val, labels=None)
            model.ae.eval() # prep model for evaluation
            for j in range(model.n_clusters):
                model.classifiers[j][0].eval()

            z_train = model(torch.Tensor(X_train).to(args.device), output="decoded")

            # evaluate clustering performance
            cluster_indices, _ = vq(z_train.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
            cluster_indices = torch.Tensor(cluster_indices).type(torch.LongTensor)
            train_sil = silhouette_new(z_train.data.cpu().numpy(), cluster_indices.data.cpu().numpy(), metric='euclidean')
            # print("Training Cluster counts")
            # print(np.bincount(cluster_indices))

            preds = torch.zeros((len(z_train), args.n_classes))

            # Calculate Training Metrics
            nmi, acc, ari = 0, 0, 0
            train_loss = 0
            B = []

            # Training loss
            for j in range(model.n_clusters):
                cluster_idx = np.where(cluster_indices == j)[0]
                X_cluster = z_train[cluster_idx]
                y_cluster = torch.Tensor(y_train[cluster_idx]).type(torch.LongTensor)
                print(np.bincount(y_cluster))

                y_clstr = label_binarize(y_train[cluster_idx], classes=list(range(args.n_classes+1)))[:,:args.n_classes]
                y_clstr = torch.Tensor(y_clstr).type(torch.LongTensor)

                cluster_dist_mat = torch.cdist(model.class_cluster_layer[j].detach(), model.class_cluster_layer[j].detach()) + U

                # point wise distanes for every cluster
                min_dists = (softmin(10*cluster_dist_mat)*cluster_dist_mat).sum(axis=1)
                # minimum inter centroid distance overall
                # csl = (torch.nn.Softmin(dim=0)(10*min_dists)*min_dists).sum()
                pt_dists = torch.cdist(X_cluster, model.class_cluster_layer[j].detach())
                # csl = torch.mean(torch.nn.Softmax(dim=1)(2*pt_dists)*y_clstr)
                csl = cac_criterion(X_cluster, y_cluster)

                # print(torch.sum(torch.nn.Softmax(dim=1)(2*pt_dists)*y_clstr, dim=1))
                # num = torch.sum((pt_dists*y_clstr), dim=1)/torch.sum(y_clstr, dim=1)
                # den = torch.sum((pt_dists*(1-y_clstr)), dim=1)/torch.sum(1-y_clstr, dim=1)
                # csl = torch.sum(torch.log(num/den))
                # csl = torch.sum(num - torch.log(den))

                dcn_loss  = args.beta*torch.linalg.norm(X_cluster - model.cluster_layer[j].detach())/len(cluster_idx)
                dcn_loss += args.eta*csl
                train_loss += dcn_loss

            # Evaluate model on Validation dataset
            z_val, x_val_bar = model(torch.FloatTensor(X_val).to(args.device), output='latent')
            cluster_ids, _ = vq(z_val.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
            cluster_ids = torch.Tensor(cluster_ids).type(torch.LongTensor)
            preds = torch.zeros((len(z_val), args.n_classes))

            # Hard Predictions
            for j in range(model.n_clusters):
                cluster_id = np.where(cluster_ids == j)[0]
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
            val_sil = silhouette_new(z_val.data.cpu().numpy(), cluster_ids.data.cpu().numpy(), metric='euclidean')
            val_feature_diff, val_WDFD = 0, 0
            # val_feature_diff = calculate_HTFD(X_val, cluster_ids)
            # val_WDFD = calculate_WDFD(X_val, cluster_ids)
            val_class_loss = torch.mean(criterion(preds, torch.Tensor(y_val).type(torch.LongTensor)))
            reconstr_loss = F.mse_loss(x_val_bar, torch.FloatTensor(X_val).to(args.device))
            val_loss = reconstr_loss
            dcn_loss = 0
            centroid_sep = 0
            for j in range(model.n_clusters):
                cluster_idx = np.where(cluster_ids == j)[0]
                X_cluster = z_val[cluster_idx]
                y_cluster = torch.Tensor(y_val[cluster_idx]).type(torch.LongTensor)
                y_clstr = label_binarize(y_val[cluster_idx], classes=list(range(args.n_classes+1)))[:,:args.n_classes]
                y_clstr = torch.Tensor(y_clstr).type(torch.LongTensor)

                cluster_dist_mat = torch.cdist(model.class_cluster_layer[j].detach(), model.class_cluster_layer[j].detach()) + U

                # point wise distanes for every cluster
                min_dists = (softmin(10*cluster_dist_mat)*cluster_dist_mat).sum(axis=1)

                # minimum inter centroid distance overall
                csl = (torch.nn.Softmin(dim=0)(10*min_dists)*min_dists).sum()
                # pt_dists = torch.cdist(X_cluster, model.class_cluster_layer[j].detach())
                pt_dists = torch.sqrt(torch.cdist(model.class_cluster_layer[j].detach(), model.class_cluster_layer[j].detach()).mean())

                # csl = torch.mean(torch.nn.Softmax(dim=1)(2*pt_dists)*y_clstr)
                csl = cac_criterion(X_cluster, y_cluster.clone().detach())

                # num = torch.sum((pt_dists*y_clstr), dim=1)/torch.sum(y_clstr, dim=1)
                # den = torch.sum((pt_dists*(1-y_clstr)), dim=1)/torch.sum(1-y_clstr, dim=1)
                # csl = torch.sum(torch.log(num/den))
                # csl = torch.sum(num - torch.log(den))
                # print("num", torch.mean(num))
                # print("den", torch.mean(den))
                n_pts = np.where(y_cluster == 0)[0]
                p_pts = np.where(y_cluster == 1)[0]
                centroid_sep += pt_dists*len(y_cluster)

                dcn_loss += args.beta*torch.linalg.norm(X_cluster - model.cluster_layer[j])/len(cluster_idx)
                dcn_loss += args.eta*csl

            centroid_sep /= len(X_val)
            q_batch = source_distribution(z_val, model.cluster_layer, alpha=model.alpha)
            P = torch.sum(torch.nn.Softmax(dim=1)(10*q_batch), axis=0)
            P = P/P.sum()
            Q = torch.ones(args.n_clusters)/args.n_clusters # Uniform distribution

            if args.cluster_balance == "kl":
                cluster_balance_loss = F.kl_div(P.log(), Q, reduction='batchmean')
            else:
                cluster_balance_loss = torch.linalg.norm(torch.sqrt(P) - torch.sqrt(Q))

            if args.delta != 0:
                val_loss += delta*cluster_balance_loss
            if args.eta != 0:
                val_loss += dcn_loss

            epoch_len = len(str(N_EPOCHS))

            print_msg = (f'\n[{epoch:>{epoch_len}}/{N_EPOCHS:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.3f} ' +
                         f'train_SIL: {train_sil:.3f} '  +
                         f'valid_loss: {val_loss:.3f} '  +
                         f'valid_F1: {val_f1:.3f} '  +
                         f'valid_AUC: {val_auc:.3f} ' + 
                         f'valid_AUPRC: {val_auprc:.3f} ' + 
                         f'valid_Feature_p: {val_feature_diff:.3f} ' + 
                         f'valid_WDFD: {val_WDFD:.3f} ' + 
                         f'valid_SIL: {val_sil:.3f} ' + 
                         f'valid_Class_Loss: {val_class_loss:.3f} ' +
                         f'valid_Centroid_Sep: {centroid_sep:.3f}')

            print(print_msg)

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            es([-val_f1, -val_auc], model)
            if es.early_stop == True:
                break

        # Normal Training
        epoch_loss = 0
        epoch_sep_loss = 0
        epoch_balance_loss = 0

        model.ae.train() # prep model for evaluation
        for j in range(model.n_clusters):
            model.classifiers[j][0].train()

        for batch_idx, (x_batch, y_batch, idx) in enumerate(train_loader):
            total_loss = 0
            x_batch = x_batch.to(device)
            idx = idx.to(device)
            delta_mu   = torch.zeros((args.n_clusters, args.latent_dim)).to(args.device)
            delta_mu   = torch.zeros((args.n_clusters, args.latent_dim)).to(args.device)
            delta_mu_c = torch.zeros((args.n_clusters, args.n_classes, args.latent_dim)).to(args.device)

            km_loss        = 0
            dcn_loss       = 0
            class_sep_loss = 0

            X_latents, x_bar = model(x_batch, output='latent')
            reconstr_loss = F.mse_loss(x_bar, x_batch)

            # classifier_labels = np.argmax(q_batch.detach().cpu().numpy(), axis=1)
            cluster_id, _ = vq(X_latents.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
            cluster_id = torch.Tensor(cluster_id).type(torch.LongTensor)

            for j in range(args.n_clusters):
                cluster_pts_idx = torch.where(cluster_id == j)[0]
                if len(cluster_pts_idx) > 0:
                    X_cluster = X_latents[cluster_pts_idx]
                    y_cluster = y_batch[cluster_pts_idx]
                    y_clstr = label_binarize(y_batch[cluster_pts_idx], classes=list(range(args.n_classes+1)))[:,:args.n_classes]
                    y_clstr = torch.Tensor(y_clstr).type(torch.LongTensor)
                    # print(len(cluster_pts_idx))
                    # create centroid distance matrix for cluster 'j'

                    # norm_vec = torch.sum(cluster_dist_mat, axis=1)
                    # print(cluster_dist_mat, norm_vec)
                    # norm_cluster_dist_mat = (cluster_dist_mat.T/norm_vec).T
                    # csl = F.kl_div(softmin(10*norm_cluster_dist_mat).log(), U, reduction='none').sum(axis=1) + 0.01
                    # csl = (csl).sum()

                    min_dists = (softmin(10*cluster_dist_mat)*cluster_dist_mat).sum()
                    csl = (torch.nn.Softmin(dim=0)(10*min_dists)*min_dists).sum()
                    pt_dists = torch.cdist(X_cluster, model.class_cluster_layer[j].detach())

                    # num = torch.sum((pt_dists*y_clstr), dim=1)/torch.sum(y_clstr, dim=1)
                    # den = torch.sum((pt_dists*(1-y_clstr)), dim=1)/torch.sum(1-y_clstr, dim=1)
                    # csl = torch.sum(torch.log(num/den))
                    # csl = torch.sum(num - torch.log(den))

                    csl = cac_criterion(X_cluster, y_cluster.clone().detach())

                    dcn_loss += args.beta*torch.linalg.norm(X_latents[cluster_pts_idx] - model.cluster_layer[j].detach())/len(cluster_pts_idx)
                    # print("csl:", csl, "min_dists:", min_dists)
                    # print(csl, dcn_loss)

                    dcn_loss += args.eta*csl
                    class_sep_loss += csl

            q_batch = source_distribution(X_latents, model.cluster_layer, alpha=model.alpha)
            P = torch.sum(torch.nn.Softmax(dim=1)(10*q_batch), axis=0)
            P = P/P.sum()
            Q = torch.ones(args.n_clusters)/args.n_clusters # Uniform distribution

            if args.cluster_balance == "kl":
                cluster_balance_loss = F.kl_div(P.log(), Q, reduction='batchmean')
            else:
                cluster_balance_loss = torch.linalg.norm(torch.sqrt(P) - torch.sqrt(Q))

            loss = reconstr_loss
            if args.delta != 0:
                loss += delta*cluster_balance_loss
            if args.eta != 0:
                loss += dcn_loss

            epoch_loss += loss
            epoch_balance_loss += cluster_balance_loss
            epoch_sep_loss += class_sep_loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Update Assignments
            X_latents, x_bar = model(x_batch, output='latent')
            # print(X_latents, model.cluster_layer)
            cluster_id, _ = vq(X_latents.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
            cluster_id = torch.Tensor(cluster_id).type(torch.LongTensor)

            # Update all the class centroids
            counter = torch.ones(args.n_clusters)*100
            class_counter = torch.ones(args.n_clusters, args.n_classes)*100
            for pt in range(len(X_latents)):
                pt_cluster = cluster_id[pt]
                eta = 1/counter[pt_cluster]
                udpated_cluster = ((1 - eta) * model.cluster_layer.data[pt_cluster] +
                               eta * X_latents.data[pt])
                model.cluster_layer.data[pt_cluster] = udpated_cluster
                counter[pt_cluster] += 1
                
                pt_class = y_batch[pt]
                eta = 1/class_counter[pt_cluster][pt_class]
                updated_class_cluster = ((1 - eta) * model.class_cluster_layer.data[pt_cluster][pt_class] +
                               eta * X_latents.data[pt])
                model.class_cluster_layer.data[pt_cluster][pt_class] = updated_class_cluster
                class_counter[pt_cluster][pt_class] += 1

        print('Epoch: {:02d} | Epoch KM Loss: {:.3f} | Total Loss: {:.3f} | Cluster Sep Loss: {:.3f}'.format(
                    epoch, dcn_loss, epoch_loss, epoch_sep_loss))
        train_losses.append([np.round(epoch_loss.item(),3), np.round(epoch_balance_loss.item(),3)])

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

    z_train = model(torch.FloatTensor(np.array(X_train)).to(args.device))
    cluster_ids_train, _ = vq(z_train.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
    cluster_ids_train = torch.Tensor(cluster_ids_train).type(torch.LongTensor)
    X_latents_data_loader = list(zip(z_train.to(args.device), y_train))

    train_loader_latents = torch.utils.data.DataLoader(X_latents_data_loader,
        batch_size=1024, shuffle=False)

    B = []

    # print(np.bincount(cluster_ids_train))
    # plot(model, torch.FloatTensor(np.array(X_train)).to(args.device), y_train, labels=cluster_ids_train)

    # Post clustering training
    for e in range(1000):
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
        es([-val_f1, -val_auc], model)
        if es.early_stop == True:
            sil_scores.append(silhouette_new(z_train.data.cpu().numpy(), cluster_ids_train.data.cpu().numpy(), metric='euclidean'))
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

    # # Evaluate model on Test dataset
    z_test = model(torch.FloatTensor(X_test).to(args.device))
    cluster_ids, _ = vq(z_test.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
    cluster_ids = torch.Tensor(cluster_ids).type(torch.LongTensor)

    test_loss = 0
    local_sum_loss = 0
    test_preds = torch.zeros((len(z_test), args.n_classes))

    # Hard local predictions
    for j in range(model.n_clusters):
        cluster_id = np.where(cluster_ids == j)[0]
        X_cluster = z_test[cluster_id]
        y_cluster = torch.Tensor(y_test[cluster_id]).type(torch.LongTensor)
        cluster_test_preds = model.classifiers[j][0](X_cluster)
        test_preds[cluster_id,:] = cluster_test_preds
        local_sum_loss += torch.sum(criterion(cluster_test_preds, y_cluster))
    
    test_metrics = performance_metrics(y_test, test_preds.detach().numpy(), args.n_classes)
    test_f1  = test_metrics['f1_score']
    test_auc = test_metrics['auroc']
    test_auprc = test_metrics['auprc']
    test_minpse = test_metrics['minpse']
    test_acc = test_metrics['acc']

    test_loss = torch.mean(criterion(test_preds, torch.Tensor(y_test).type(torch.LongTensor)))
    local_sum_loss /= len(X_test)

    test_losses.append(test_loss.item())
    local_sum_test_losses.append(local_sum_loss.item())

    print("Run #{}".format(r))
    print('Loss Metrics - Test Loss {:.3f}, Local Sum Test Loss {:.3f}'.format(test_loss, local_sum_loss))

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

enablePrint()

print("\n")
print("Experiment ", iteration_name)
print(iter_array)
print("Test F1: ", f1_scores)
print("Test AUC: ", auc_scores)

print("Sil scores: ", sil_scores)
print("HTFD: ", HTFD_scores)

print("Train Cluster Counts: ", np.bincount(cluster_ids_train))

# print("Train Loss: ", train_losses)

# print("Test Loss: ", test_losses)
# print("Local Test Loss: ", local_sum_test_losses)

# print("Model Complexity: ", model_complexity)

# print('Dataset\tk')
# print("{}\t{}\n".format(args.dataset, args.n_clusters))

print("[Avg]\tDataset\tk\tF1\tAUC\tAUPRC\tMINPSE\tACC\tSIL\tHTFD")

print("\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format\
    (args.dataset, args.n_clusters, np.average(f1_scores), np.average(auc_scores),\
    np.average(auprc_scores), np.average(minpse_scores), np.average(acc_scores),\
    np.average(sil_scores), np.average(HTFD_scores)))

print("[Std]\tF1\tAUC\tAUPRC\tMINPSE\tACC\tSIL\tHTFD")

print("\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format\
    (np.std(f1_scores), np.std(auc_scores),np.std(auprc_scores),\
    np.std(minpse_scores), np.std(acc_scores), np.std(sil_scores),\
    np.std(HTFD_scores)))

print("Train Loss\tTest Loss")

print("{:.3f}\t{:.3f}\n".format\
    (np.average(train_losses), np.average(test_losses)))

if args.cluster_analysis == "True":
    # WDFD_Cluster_Analysis(torch.Tensor(X_train), cluster_ids_train, column_names)
    # HTFD_Cluster_Analysis(torch.Tensor(X_train), cluster_ids_train, column_names)
    HTFD_Single_Cluster_Analysis(X_train, y_train, cluster_ids_train, column_names)