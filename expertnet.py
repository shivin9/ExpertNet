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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, label_binarize

import argparse
import numpy as np
from scipy.cluster.vq import vq, kmeans2
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.nn import Linear
from pytorchtools import EarlyStoppingEN

import numbers
from sklearn.metrics import davies_bouldin_score as dbs, adjusted_rand_score as ari
from matplotlib import pyplot as plt
color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

from models import ExpertNet,  target_distribution, source_distribution
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
parser.add_argument("--attention", default=11, type=int)
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
parser.add_argument('--pretrain_path', default= '/Users/shivin/Document/NUS/Research/CAC/CAC_DL/ExpertNet/pretrained_model/EN')
# parser.add_argument('--pretrain_path', default= '/home/shivin/CAC_code/data')

parser = parser.parse_args()  
args = parameters(parser)
base_suffix = ""

# for key in ['n_clusters', 'alpha', 'beta', 'gamma', 'delta', 'eta', 'attention']:
#     print(key, args.__dict__[key])

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

stu_f1_scores, stu_auc_scores, stu_auprc_scores, stu_acc_scores, stu_minpse_scores = [], [], [], [], [] #Attentive test results
stu_sil_scores, stu_wdfd_scores, stu_HTFD_scores, stu_w_HTFD_scores = [], [], [], []

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
    scale, column_names, train_data, val_data, test_data = get_train_val_test_loaders(args, r_state=r, n_features=args.n_features)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    print("Stats: ", (sum(y_train)+sum(y_val)+sum(y_test)), (len(y_train)+len(y_val)+len(y_test)))

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

    if args.expt == 'ExpertNet':
        ae_layers = [128, 64, args.n_z, 64, 128]
        expert_layers = [args.n_z, 128, 64, 32, 16, args.n_classes]

    else:
        # DeepCAC expts
        ae_layers = [64, args.n_z, 64]
        expert_layers = [args.n_z, 30, args.n_classes]

    print(args.input_dim, args.n_features)
    model = ExpertNet(
            ae_layers,
            expert_layers,
            args.lr_enc,
            args.lr_exp,
            args=args).to(args.device)

    model.pretrain(train_loader, args.pretrain_path)

    # Initiate cluster parameters
    device = args.device
    y = y_train
    x_bar, hidden = model.ae(torch.Tensor(X_train).to(args.device))
    original_cluster_centers, cluster_indices = kmeans2(hidden.data.cpu().numpy(), k=args.n_clusters, minit='++')
    model.cluster_layer.data = torch.tensor(original_cluster_centers).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')

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
    es = EarlyStoppingEN(dataset=suffix, patience=7)
    train_losses, e_train_losses = [], []

    for epoch in range(N_EPOCHS):
        beta = args.beta
        gamma = args.gamma
        delta = args.delta
        eta = args.eta
        if epoch % args.log_interval == 0:

            # if args.plot == 'True':
            #     plot(model, torch.FloatTensor(X_train).to(args.device), y_train, args, labels=cluster_indices, epoch=epoch)

            model.ae.eval() # prep model for evaluation
            for j in range(model.n_clusters):
                model.classifiers[j][0].eval()

            z_train, _, q_train = model.encoder_forward(torch.Tensor(X_train).to(args.device), output="decoded")
            p_train = target_distribution(q_train.detach())

            # evaluate clustering performance
            cluster_indices = q_train.detach().cpu().numpy().argmax(1)
            preds = torch.zeros((len(z_train), args.n_classes))

            # Calculate Training Metrics
            nmi, acc, ari = 0, 0, 0
            train_loss = 0
            B = []

            for j in range(model.n_clusters):
                cluster_idx = np.where(cluster_indices == j)[0]
                X_cluster = z_train[cluster_idx]
                y_cluster = torch.Tensor(y_train[cluster_idx]).type(torch.LongTensor).to(model.device)

                # B.append(torch.max(torch.linalg.norm(X_cluster, axis=1), axis=0).values)
                cluster_preds = model.classifiers[j][0](X_cluster)
                train_loss += torch.sum(criterion(cluster_preds, y_cluster))

            train_loss /= len(z_train)

            # Evaluate model on Validation dataset
            z_val, x_val_bar, q_val = model.encoder_forward(torch.FloatTensor(X_val).to(args.device), output="decoded")
            cluster_ids = torch.argmax(q_val, axis=1)

            preds = model.predict(torch.FloatTensor(X_val).to(args.device))

            # Classification Matrics
            val_metrics = performance_metrics(y_val, preds.detach().numpy(), args.n_classes)
            val_f1  = val_metrics['f1_score']
            val_auc = val_metrics['auroc']
            val_auprc = val_metrics['auprc']
            val_minpse = val_metrics['minpse']

            # Clustering Metrics
            val_sil = silhouette_new(z_val.data.cpu().numpy(), cluster_ids.data.cpu().numpy(), metric='euclidean')
            val_feature_diff, val_WDFD = 0, 0
            val_feature_diff = calculate_HTFD(X_val, cluster_ids)
            val_WDFD = calculate_WDFD(X_val, cluster_ids)

            q_val = source_distribution(z_val, model.cluster_layer, alpha=model.alpha)
            P = torch.sum(torch.nn.Softmax(dim=1)(10*q_val), axis=0)
            P = P/P.sum()
            Q = torch.ones(args.n_clusters)/args.n_clusters # Uniform distribution

            if args.cluster_balance == "kl":
                val_cluster_balance_loss = F.kl_div(P.log(), Q, reduction='batchmean')
            else:
                val_cluster_balance_loss = torch.linalg.norm(torch.sqrt(P) - torch.sqrt(Q))

            p_val = target_distribution(q_val.detach())
            val_reconstr_loss = F.mse_loss(x_val_bar, torch.FloatTensor(X_val).to(args.device))
            val_km_loss = F.kl_div(q_val.log(), p_val, reduction='batchmean')
            val_class_loss = torch.mean(criterion(preds, torch.Tensor(y_val).type(torch.LongTensor)))

            val_loss  = args.alpha*val_reconstr_loss
            val_loss += args.beta*val_km_loss
            val_loss += args.gamma*val_class_loss
            val_loss += args.delta*val_cluster_balance_loss

            epoch_len = len(str(N_EPOCHS))

            print_msg = (f'\n[{epoch:>{epoch_len}}/{N_EPOCHS:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.3f} ' +
                         f'valid_loss: {val_loss:.3f} '  +
                         f'valid_F1: {val_f1:.3f} '  +
                         f'valid_AUC: {val_auc:.3f} ' + 
                         f'valid_AUPRC: {val_auprc:.3f} ' + 
                         f'valid_MINPSE: {val_minpse:.3f} ' + 
                         f'valid_Feature_p: {val_feature_diff:.3f} ' + 
                         f'valid_WDFD: {val_WDFD:.3f} ' + 
                         f'valid_Silhouette: {val_sil:.3f}')

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
        epoch_balance_loss = 0
        epoch_class_loss = 0
        epoch_km_loss = 0
        
        model.ae.train() # prep model for evaluation
        for j in range(model.n_clusters):
            model.classifiers[j][0].train()

        for batch_idx, (x_batch, y_batch, idx) in enumerate(train_loader):
            total_loss = 0
            x_batch = x_batch.to(device)
            idx = idx.to(device)

            X_latents, x_bar, q_batch = model.encoder_forward(x_batch)
            reconstr_loss = F.mse_loss(x_bar, x_batch)

            sub_epochs = min(10, 1 + int(epoch/5))
            # sub_epochs = 10

            for _ in range(sub_epochs):
                model.expert_forward(x_batch, y_batch, X_latents, q_batch, backprop_enc=False, backprop_local=True, attention=attention_train)

            q, class_loss = model.expert_forward(x_batch, y_batch, X_latents, q_batch, backprop_enc=True, backprop_local=False, attention=attention_train)

            class_loss /= len(X_latents)

            q_batch = source_distribution(X_latents, model.cluster_layer, alpha=model.alpha)
            P = torch.sum(torch.nn.Softmax(dim=1)(10*q_batch), axis=0)
            P = P/P.sum()
            Q = torch.ones(args.n_clusters)/args.n_clusters # Uniform distribution

            if args.cluster_balance == "kl":
                cluster_balance_loss = F.kl_div(P.log(), Q, reduction='batchmean')
            else:
                cluster_balance_loss = torch.linalg.norm(torch.sqrt(P) - torch.sqrt(Q))

            km_loss = F.kl_div(q_batch.log(), p_train[idx], reduction='batchmean')

            loss = args.alpha*reconstr_loss
            if args.beta != 0:
                loss += beta*km_loss
            if args.gamma != 0:
                loss += gamma*class_loss
            if args.delta != 0:
                loss += delta*cluster_balance_loss

            epoch_loss += loss
            epoch_class_loss += class_loss
            epoch_balance_loss += cluster_balance_loss
            epoch_km_loss += km_loss
            model.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            model.optimizer.step()

        print('Epoch: {:02d} | Epoch KM Loss: {:.3f} | Total Loss: {:.3f} | Classification Loss: {:.3f} | Cluster Balance Loss: {:.3f}'\
        .format(epoch, epoch_km_loss, epoch_loss, epoch_class_loss, loss))
        train_losses.append([np.round(epoch_loss.item(),3), np.round(epoch_class_loss.item(),3)])

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
    es = EarlyStoppingEN(dataset=suffix)

    q_train, z_train = model.encoder_forward(torch.FloatTensor(np.array(X_train)).to(args.device), output="latent")
    cluster_id_train = torch.argmax(q_train, axis=1)

    X_latents_data_loader = list(zip(z_train.to(args.device),q_train, y_train))

    train_loader_latents = torch.utils.data.DataLoader(X_latents_data_loader,
        batch_size=1024, shuffle=False)

    B = []

    print(np.bincount(cluster_id_train))
    # plot(model, torch.FloatTensor(np.array(X_train)).to(args.device), y_train,\
         # torch.FloatTensor(np.array(X_test)).to(args.device), y_test)

    # Post clustering training
    for e in range(N_EPOCHS):
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1 = 0
        acc = 0

        for j in range(model.n_clusters):
            model.classifiers[j][0].train()

        # Full training of local networks
        for batch_idx, (X_latents, q_batch, y_batch) in enumerate(train_loader_latents):
            _, total_loss = model.expert_forward(x_batch, y_batch, X_latents, q_batch, backprop_enc=False, backprop_local=True)

        for j in range(model.n_clusters):
            model.classifiers[j][0].eval()

        # Evaluate model on Validation set
        q_val, z_val = model.encoder_forward(torch.FloatTensor(X_val).to(args.device), output="latent")
        cluster_ids_val = torch.argmax(q_val, axis=1)

        preds = model.predict(torch.FloatTensor(X_val).to(args.device))

        val_metrics = performance_metrics(y_val, preds.detach().numpy(), args.n_classes)
        val_f1  = val_metrics['f1_score']
        val_auc = val_metrics['auroc']
        val_auprc = val_metrics['auprc']
        val_minpse = val_metrics['minpse']

        val_sil = silhouette_new(z_val.data.cpu().numpy(), cluster_ids_val.data.cpu().numpy(), metric='euclidean')
        val_loss = torch.mean(criterion(preds, torch.Tensor(y_val).type(torch.LongTensor)))
        epoch_len = len(str(N_EPOCHS))
        
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
        cluster_ids_train = torch.argmax(q_train, axis=1)

        if args.optimize == 'auc':
            opt = val_auc
        elif args.optimize == 'auprc':
            opt = val_auprc
        else:
            opt = -val_loss
        es([val_f1, opt], model)

        if es.early_stop == True or epoch == N_EPOCHS - 1:
            # train_losses.append(train_loss.item())
            sil_scores.append(silhouette_new(z_train.data.cpu().numpy(), cluster_ids_train.data.cpu().numpy(), metric='euclidean'))
            HTFD_scores.append(calculate_HTFD(X_train, cluster_ids_train))
            wdfd_scores.append(calculate_WDFD(X_train, cluster_ids_train))
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

    model.ae.eval() # prep model for evaluation
    for j in range(model.n_clusters):
        model.classifiers[j][0].eval()

    # Evaluate model on Test dataset
    q_test, z_test = model.encoder_forward(torch.FloatTensor(X_test).to(args.device), output="latent")
    cluster_ids = torch.argmax(q_test, axis=1)

    X_latents_data_loader = list(zip(z_test.to(args.device), q_test, y_test))

    test_loader_latents = torch.utils.data.DataLoader(X_latents_data_loader,
        batch_size=1024, shuffle=False)


    test_loss = 0
    e_test_loss = 0
    local_sum_loss = 0

    test_preds_e = model.predict(torch.FloatTensor(X_test).to(args.device), attention=attention_test)
    e_test_loss = torch.mean(criterion(test_preds_e, torch.Tensor(y_test).type(torch.LongTensor)))

    test_metrics = performance_metrics(y_test, test_preds_e.detach().numpy(), args.n_classes)
    e_test_f1  = test_metrics['f1_score']
    e_test_auc = test_metrics['auroc']
    e_test_auprc = test_metrics['auprc']
    e_test_minpse = test_metrics['minpse']
    e_test_acc = test_metrics['acc']
    e_test_HTFD = calculate_HTFD(X_test, cluster_ids)

    test_preds = model.predict(torch.FloatTensor(X_test).to(args.device), attention=False)
    test_loss = torch.mean(criterion(test_preds, torch.Tensor(y_test).type(torch.LongTensor)))

    test_metrics = performance_metrics(y_test, test_preds.detach().numpy(), args.n_classes)
    test_f1  = test_metrics['f1_score']
    test_auc = test_metrics['auroc']
    test_auprc = test_metrics['auprc']
    test_minpse = test_metrics['minpse']
    test_acc = test_metrics['acc']
    

    test_losses.append(test_loss.item())
    e_test_losses.append(e_test_loss.item())

    print("Run #{}".format(r))
    print('Loss Metrics - Test Loss {:.3f}, E-Test Loss {:.3f}, Local Sum Test Loss {:.3f}'.format(test_loss, e_test_loss, local_sum_loss))

    print('Clustering Metrics - Acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi),\
          ', ari {:.4f}, HTFD {:.3f}'.format(ari, e_test_HTFD))

    print('Classification Metrics - Test F1 {:.3f}, Test AUC {:.3f}, Test AUPRC {:.3f}, Test MIN_PSE {:.3f}'.format(test_f1,\
        test_auc, test_auprc, test_minpse), ', E-Test F1 {:.3f}, E-Test AUC {:.3f}, E-Test AUPRC {:.3f}, E-Test MIN_PSE {:.3f}'.format\
        (e_test_f1, e_test_auc, e_test_auprc, e_test_minpse))

    print("\n")

    f1_scores.append(test_f1)
    auc_scores.append(test_auc)
    auprc_scores.append(test_auprc)
    minpse_scores.append(test_minpse)
    acc_scores.append(test_acc)

    e_f1_scores.append(e_test_f1)
    e_auc_scores.append(e_test_auc)
    e_auprc_scores.append(e_test_auprc)
    e_minpse_scores.append(e_test_minpse)
    e_acc_scores.append(e_test_acc)

    ####################################################################################
    ####################################################################################
    ####################################################################################
    ################################### Feature Imp. ###################################
    ####################################################################################
    ####################################################################################
    ####################################################################################

    encoder_reg = RandomForestClassifier(random_state=0)
    regs = [RandomForestClassifier(random_state=0) for _ in range(args.n_clusters)]

    _ , z_train = model.encoder_forward(torch.FloatTensor(X_train).to(args.device), output='latent')
    cluster_ids, _ = vq(z_train.data.cpu().numpy(), model.cluster_layer.cpu().numpy())
    cluster_ids = torch.Tensor(cluster_ids).type(torch.LongTensor)

    train_preds = torch.zeros((len(z_train), args.n_classes))
    feature_importances = np.zeros((args.n_clusters, args.input_dim))

    encoder_reg.fit(X_train, cluster_ids.numpy())

    # Weighted predictions... should be without attention only
    for j in range(model.n_clusters):
        cluster_id = torch.where(cluster_ids == j)[0]
        X_cluster = z_train[cluster_id]
        cluster_preds = model.classifiers[j][0](X_cluster)
        train_preds[cluster_id,:] = cluster_preds

        # train student on teacher's predictions
        y_cluster = np.argmax(train_preds[cluster_id].detach().numpy(), axis=1)
        
        # train student on real labels
        y_cluster_true = y_train[cluster_id]

        # Train the local regressors on the data embeddings
        # Some test data might not belong to any cluster
        if len(cluster_id) > 0:
            regs[j].fit(X_train[cluster_id], y_cluster_true)
            best_features = np.argsort(regs[j].feature_importances_)[::-1][:10]
            feature_importances[j,:] = regs[j].feature_importances_
            print("Cluster # ", j, "sized: ", len(cluster_id), "label distr: ", np.bincount(y_cluster_true)/len(y_cluster_true))
            print(list(zip(column_names[best_features], np.round(regs[j].feature_importances_[best_features], 3))))
            print("=========================\n")

    # Testing performance of downstream classifier on cluster embeddings
    _, z_test = model.encoder_forward(torch.FloatTensor(X_test).to(args.device), output='latent')
    cluster_ids_test_model, _ = vq(z_test.data.cpu().numpy(), model.cluster_layer.cpu().numpy())

    cluster_ids_test = encoder_reg.predict(X_test)
    cluster_ids_test = torch.Tensor(cluster_ids_test).type(torch.LongTensor)
        
    test_preds = torch.zeros((len(X_test), args.n_classes))
    y_pred = np.zeros((len(X_test), args.n_classes))

    for j in range(model.n_clusters):
        cluster_id = torch.where(cluster_ids_test == j)[0]
        X_cluster = X_test[cluster_id]

        # Some test data might not belong to any cluster
        if len(cluster_id) > 0:
            y_pred[cluster_id] = regs[j].predict_proba(X_cluster)

    test_metrics = performance_metrics(y_test, y_pred, args.n_classes)
    test_f1  = test_metrics['f1_score']
    test_auc = test_metrics['auroc']
    test_auprc = test_metrics['auprc']
    test_minpse = test_metrics['minpse']
    test_acc = test_metrics['acc']

    stu_f1_scores.append(test_f1)
    stu_auc_scores.append(test_auc)
    stu_auprc_scores.append(test_auprc)
    stu_minpse_scores.append(test_minpse)

    print('Student Network Classification Metrics - Test F1 {:.3f}, Test AUC {:.3f},\
        Test AUPRC {:.3f}'.format(test_f1, test_auc, test_auprc))

    print("\n")

    feature_diff = 0
    cntr = 0
    for i in range(args.n_clusters):
        for j in range(args.n_clusters):
            if i > j:
                ci = torch.where(cluster_ids == i)[0]
                cj = torch.where(cluster_ids == j)[0]
                Xi = X_train[ci]
                Xj = X_train[cj]
                feature_diff += 100*sum(feature_importances[i]*feature_importances[j]*(ttest_ind(Xi, Xj, axis=0)[1] < 0.05))/args.input_dim
                # print("Cluster [{}, {}] p-value: ".format(i,j), feature_diff)
                cntr += 1

    if cntr == 0:
        w_HTFD_scores.append(0)
    else:
        print("Average Feature Difference: ", feature_diff/cntr)
        w_HTFD_scores.append(feature_diff/cntr)


# print("\n")
# print("Experiment ", iteration_name)
# print(iter_array)
# print("Test F1: ", e_f1_scores)
# print("Test AUC: ", e_auc_scores)
# print("Test AUPRC: ", e_auprc_scores)
# print("Test MINPSE: ", e_minpse_scores)

# print("Sil scores: ", sil_scores)
# print("HTFD: ", HTFD_scores)
# print("WDFD: ", wdfd_scores)

# # print("Train Loss: ", train_losses)
# # print("E-Train Loss: ", e_train_losses)

# print("Test Loss: ", test_losses)
# print("E-Test Loss: ", e_test_losses)
# print("Local Test Loss: ", local_sum_test_losses)

# print("Model Complexity: ", model_complexity)

enablePrint()

for key in ['alpha', 'beta', 'gamma', 'delta']:
    print(key, args.__dict__[key])

print("[Avg]\tDataset\tk\tF1\tAUC\tAUPRC\tMINPSE\tACC")
print("\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(args.dataset, args.n_clusters,\
    np.avg(e_f1_scores), np.avg(e_auc_scores), np.avg(e_auprc_scores), np.avg(e_minpse_scores), np.avg(e_acc_scores)))

print("[Std]\tF1\tAUC\tAUPRC\tMINPSE\tE-ACC")
print("\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format\
    (np.std(e_f1_scores), np.std(e_auc_scores), np.std(e_auprc_scores), np.std(e_minpse_scores), np.std(e_acc_scores)))

print('[Avg]\tTr-SIL\tHTFD\tWDFD')
print("\t{:.3f}\t{:.3f}\t{:.3f}\n".format(np.avg(sil_scores),\
    np.avg(HTFD_scores), np.avg(wdfd_scores)))

print('[Std]\tTr-SIL\tHTFD\tWDFD')
print("\t{:.3f}\t{:.3f}\t{:.3f}\n".format(np.std(sil_scores),\
    np.std(HTFD_scores), np.std(wdfd_scores)))

print("Distilled Model Metrics")
print("[Avg]\tDataset\tk\tF1\tAUC\tAUPRC\tMINPSE")
print("\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(args.dataset, args.n_clusters,\
    np.avg(stu_f1_scores), np.avg(stu_auc_scores), np.avg(stu_auprc_scores), np.avg(stu_minpse_scores)))

print("[Std]\tF1\tAUC\tAUPRC\tMINPSE")
print("\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format\
    (np.std(stu_f1_scores), np.std(stu_auc_scores), np.std(stu_auprc_scores), np.std(stu_minpse_scores)))


# print("F1\tAUC\tAUPRC\tACC")

# print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format\
#     (np.avg(f1_scores), np.avg(auc_scores), np.avg(auprc_scores), np.avg(acc_scores)))

# print("Train Loss\tE-Train Loss\tTest Loss\tE-Test Loss")

# print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format\
#     (np.avg(train_losses), np.avg(e_train_losses),\
#     np.avg(test_losses), np.avg(e_test_losses)))

if args.cluster_analysis == "True":
    _ , z_train = model.encoder_forward(torch.FloatTensor(X_train).to(args.device), output='latent')
    plot(model, torch.FloatTensor(X_train).to(args.device), y_train, args, X_latents=z_train, labels=cluster_indices, epoch=epoch)
    WDFD_Single_Cluster_Analysis(X_train, y_train, cluster_ids_train, column_names,\
        scale=scale, X_latents=z_train, dataset=args.dataset, n_results=15)

    HTFD_Single_Cluster_Analysis(scale.inverse_transform(X_train), y_train, cluster_ids_train, column_names,\
        scale=None, n_results=25)