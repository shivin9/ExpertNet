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

import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
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

from models import MultiHeadIDEC,  target_distribution
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
        self.log_interval = parser.log_interval
        self.pretrain_path = parser.pretrain_path + "/" + self.dataset + ".pth"

args = parameters(parser)

for key in args.__dict__:
    print(key, args.__dict__[key])

train_data, val_data, test_data = get_train_val_test_loaders(args)
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
y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
cluster_indices = kmeans.labels_
original_cluster_centers = kmeans.cluster_centers_

nmi = 0
for j in range(args.n_clusters):
    kmeans = KMeans(n_clusters=args.n_classes, n_init=20)
    cluster_idx = np.where(cluster_indices == j)[0]
    y_pred_idx = kmeans.fit_predict(hidden.data.cpu().numpy()[cluster_idx])
    nmi_k = nmi_score(y_pred_idx, y[cluster_idx])
    nmi += len(cluster_idx)*nmi_k

print("NMI score={:.4f}".format(nmi/len(X_train)))

y_pred_last = y_pred
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
es = EarlyStoppingCAC(dataset = args.dataset)

for epoch in range(N_EPOCHS):
    if epoch % args.log_interval == 0:
        # plot(model, torch.FloatTensor(X_train).to(args.device), y_train)
        X_latents, _, tmp_q = model(torch.Tensor(X_train).to(args.device), output="decoded")
        tmp_q, tmp_q_p, tmp_q_n = tmp_q
        # update target distribution p
        tmp_q = tmp_q.data

        p = target_distribution(tmp_q)
        p_p = target_distribution(tmp_q_p)
        p_n = target_distribution(tmp_q_n)

        # evaluate clustering performance
        y_pred = tmp_q.cpu().numpy().argmax(1)
        delta_label = np.sum(y_pred != y_pred_last).astype(
            np.float32) / y_pred.shape[0]
        y_pred_last = y_pred

        # Calculate Training Metrics
        nmi, acc, ari = 0, 0, 0
        train_loss = 0
        for j in range(args.n_clusters):
            kmeans = KMeans(n_clusters=args.n_classes, n_init=20)
            cluster_idx = np.where(cluster_indices == j)[0]
            y_pred_idx = kmeans.fit_predict(X_latents.data.cpu().numpy()[cluster_idx])
            nmi_k = nmi_score(y_pred_idx, y[cluster_idx])
            nmi += nmi_k * len(cluster_idx)/len(X_train)
            acc += cluster_acc(y_pred_idx, y[cluster_idx]) * len(cluster_idx)/len(X_train)
            ari += ari_score(y_pred_idx, y[cluster_idx]) * len(cluster_idx)/len(X_train)

            X_cluster = X_latents[cluster_idx]
            y_cluster = torch.Tensor(y_train[cluster_idx]).type(torch.LongTensor).to(model.device)

            classifier_k, optimizer_k = model.classifiers[j]
            y_pred_cluster = classifier_k(X_cluster)
            cluster_los = criterion(y_pred_cluster, y_cluster)
            train_loss += cluster_los

        model.ae.eval() # prep model for evaluation
        for j in range(model.n_clusters):
            model.classifiers[j][0].eval()

        # Evaluate model on Test dataset
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

    # Normal Training
    epoch_loss = 0
    epoch_sep_loss = 0

    model.ae.train() # prep model for evaluation
    for j in range(model.n_clusters):
        model.classifiers[j][0].train()

    for batch_idx, (x_batch, y_batch, idx) in enumerate(train_loader):
        # torch.autograd.set_detect_anomaly(True)
        x_batch = x_batch.to(device)
        idx = idx.to(device)

        X_latents, x_bar, qs = model(x_batch)
        q, q_p, q_n = qs
        reconstr_loss = F.mse_loss(x_bar, x_batch)

        classifier_labels = np.zeros(len(idx))
        train_epochs = min(10, 1 + int(epoch/5))
        for _ in range(train_epochs):
            # Choose classifier for a point probabilistically
            for j in range(len(idx)):
                classifier_labels[j] = np.random.choice(range(args.n_clusters), p = q[j].detach().numpy())

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

        N1 = sum(y_batch).item()
        N0 = len(y_batch) - N1

        p_idx = torch.where(y_batch == 1)[0]
        n_idx = torch.where(y_batch == 0)[0]
        
        delta_mu_p = torch.zeros((args.n_clusters, args.latent_dim)).to(args.device)
        delta_mu_n = torch.zeros((args.n_clusters, args.latent_dim)).to(args.device)
        delta_mu   = torch.zeros((args.n_clusters, args.latent_dim)).to(args.device)
        cluster_id = torch.argmax(q, 1)
        
        positive_class_dist = 0
        negative_class_dist = 0
        km_loss             = 0
        class_sep_loss = 0

        for j in range(args.n_clusters):
            pts_index = np.where(cluster_id == j)[0]
            cluster_pts = X_latents[pts_index]
            n_class_index = np.where(y[pts_index] == 0)[0]
            p_class_index = np.where(y[pts_index] == 1)[0]

            n_class = cluster_pts[n_class_index]
            p_class = cluster_pts[p_class_index]

            delta_mu_p[j,:] = p_class.sum(axis=0)/(1+len(p_class))
            delta_mu_n[j,:] = n_class.sum(axis=0)/(1+len(n_class))
            delta_mu[j,:]   = cluster_pts.sum(axis=0)/(1+len(cluster_pts))

            s1 = torch.linalg.vector_norm(X_latents[p_class_index] - model.p_cluster_layer[j])/(1+len(p_class))
            s2 = torch.linalg.vector_norm(X_latents[n_class_index] - model.n_cluster_layer[j])/(1+len(n_class))
            m12 = torch.linalg.vector_norm(model.p_cluster_layer[j] - model.n_cluster_layer[j])

            class_sep_loss += (s1+s2)/m12
            km_loss += torch.linalg.vector_norm(X_latents[pts_index] - model.cluster_layer[j])/(1+len(cluster_pts))

        loss = reconstr_loss
        if args.beta != 0:
            loss += args.beta*km_loss
        if args.gamma != 0:
            loss += args.gamma*class_loss
        if args.delta != 0:
            loss += args.delta*class_sep_loss

        epoch_loss += loss
        epoch_sep_loss += class_sep_loss.item()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # Update the positive and negative centroids
        for j in range(args.n_clusters):
            pts_index = np.where(cluster_id == j)[0]
            n_class_index = np.where(y[pts_index] == 0)[0]
            p_class_index = np.where(y[pts_index] == 1)[0]

            N  = len(pts_index)
            Np = len(p_class_index)
            Nn = len(n_class_index)
            model.p_cluster_layer.data[j:] -= (1/(100+Np))*delta_mu_p[j:]
            model.n_cluster_layer.data[j:] -= (1/(100+Nn))*delta_mu_n[j:]
            model.cluster_layer.data[j:]   -= (1/(100+N))*delta_mu[j:]

    print('Epoch: {:02d} | Loss: {:.3f} | Classification Loss: {:.3f} | Class Sep Loss: {:.3f}'.format(
                epoch, epoch_loss, class_loss/train_epochs, class_sep_loss))

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

es = EarlyStoppingCAC(dataset = args.dataset)

qs, latents_X = model(torch.FloatTensor(np.array(X_train)).to(args.device), output="latent")
q_train = qs[0]
cluster_id_train = torch.argmax(q_train, axis=1)

# X_latents_data_loader = list(zip(latents_X, cluster_id_train, y_train))
X_latents_data_loader = list(zip(latents_X.to(args.device),q_train, y_train))

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

    model.ae.train() # prep model for evaluation
    for j in range(model.n_clusters):
        model.classifiers[j][0].train()

    # Full training of local networks
    for batch_idx, (X_latents, q_batch, y_batch) in enumerate(train_loader_latents):
        # torch.autograd.set_detect_anomaly(True)

        classifier_labels = np.zeros(len(X_latents))
        # Choose classifier for a point probabilistically
        for j in range(len(X_latents)):
            classifier_labels[j] = np.random.choice(range(args.n_clusters), p = q_batch[j].detach().numpy())

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
    
    model.ae.eval() # prep model for evaluation
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
print("Evaluating Test Data")

# Load best model trained from local training phase
model = es.load_checkpoint(model)

# # Evaluate model on Test dataset
qs, z_test = model(torch.FloatTensor(X_test).to(args.device), output="latent")
q_test = qs[0]
cluster_ids = torch.argmax(q_test, axis=1)
preds = torch.zeros((len(z_test), 2))

# Weighted predictions
for j in range(model.n_clusters):
    cluster_id = np.where(cluster_ids == j)[0]
    # X_cluster = z_test[cluster_id]
    X_cluster = z_test
    cluster_preds = model.classifiers[j][0](X_cluster)
    # print(q_test, cluster_preds[:,0])
    preds[:,0] += q_test[:,j]*cluster_preds[:,0]
    preds[:,1] += q_test[:,j]*cluster_preds[:,1]

e_test_f1 = f1_score(y_test, np.argmax(preds.detach().numpy(), axis=1))
e_test_auc = roc_auc_score(y_test, preds[:,1].detach().numpy())

# Hard local predictions
for j in range(model.n_clusters):
    cluster_id = np.where(cluster_ids == j)[0]
    X_cluster = z_test[cluster_id]
    cluster_preds = model.classifiers[j][0](X_cluster)
    preds[cluster_id,:] = cluster_preds

test_f1 = f1_score(y_test, np.argmax(preds.detach().numpy(), axis=1))
test_auc = roc_auc_score(y_test, preds[:,1].detach().numpy())

print('Iter {}'.format(e), ': Acc {:.4f}'.format(acc),
      ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

# print(test_f1, test_auc)
print('Iter {}'.format(e),', Test F1 {:.3f}, Test AUC {:.3f}'.format(test_f1, test_auc),\
    ', E-Test F1 {:.3f}, E-Test AUC {:.3f}'.format(e_test_f1, e_test_auc))