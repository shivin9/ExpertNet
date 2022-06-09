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
from pytorchtools import EarlyStoppingEN

import numbers
from sklearn.metrics import davies_bouldin_score as dbs, adjusted_rand_score as ari
from matplotlib import pyplot as plt
color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

from models import ExpertNet,  target_distribution, source_distribution
from utils import *

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default= 'aki_new')
parser.add_argument('--input_dim', default= '-1')

# Training parameters
parser.add_argument('--lr_exp', default= 0.002, type=float)
parser.add_argument('--lr_enc', default= 0.002, type=float)
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
parser.add_argument('--optimize', default='auc')

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


def train_expertnet(config, args=args):
    scale, column_names, train_data, val_data, test_data = get_train_val_test_loaders(args, r_state=0)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    train_loader = generate_data_loaders(X_train, y_train, args.batch_size)
    val_loader = generate_data_loaders(X_val, y_val, args.batch_size)
    test_loader = generate_data_loaders(X_test, y_test, args.batch_size)

    if args.verbose == 'False':
        blockPrint()

    suffix = base_suffix + "_" + iteration_name + "_" + str(0)
    n_clusters = config['n_clusters']

    if args.expt == 'ExpertNet':
        ae_layers = [64, 32, args.n_z, 32, 64]
        expert_layers = [args.n_z, 64, 32, 16, 8, args.n_classes]

    else:
        # DeepCAC expts
        ae_layers = [64, args.n_z, 64]
        expert_layers = [args.n_z, 30, args.n_classes]

    args.n_clusters = n_clusters

    model = ExpertNet(
            ae_layers,
            expert_layers,
            config['lr_exp'],
            config['lr_enc'],
            args=args).to(args.device)

    model.pretrain(train_loader, args.pretrain_path)


    # Initiate cluster parameters
    device = args.device
    y = y_train
    x_bar, hidden = model.ae(torch.Tensor(X_train).to(args.device))
    original_cluster_centers, cluster_indices = kmeans2(hidden.data.cpu().numpy(), k=n_clusters, minit='++')
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
    best_val_auprc = 0
    best_val_auc = 0
    best_val_loss = 10000
    best_model = model

    for epoch in range(N_EPOCHS):
        beta = config['beta']
        gamma = config['gamma']
        delta = config['delta']

        if epoch % args.log_interval == 0:

            if args.plot == 'True':
                plot(model, torch.FloatTensor(X_train).to(args.device), y_train, args, labels=cluster_indices, epoch=epoch)

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
            q_val, z_val = model.encoder_forward(torch.FloatTensor(X_val).to(args.device), output="latent")
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
            val_loss = torch.mean(criterion(preds, torch.Tensor(y_val).type(torch.LongTensor)))

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

            # print(print_msg)

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model = model
            # es([val_f1, val_auprc], model)
            # if es.early_stop == True:
            #     break

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

            for _ in range(sub_epochs):
                model.expert_forward(x_batch, y_batch, X_latents, q_batch, backprop_enc=False, backprop_local=True, attention=args.attention)

            q, class_loss = model.expert_forward(x_batch, y_batch, X_latents, q_batch, backprop_enc=True, backprop_local=False, attention=args.attention)

            class_loss /= len(X_latents)

            q_batch = source_distribution(X_latents, model.cluster_layer, alpha=model.alpha)
            P = torch.sum(torch.nn.Softmax(dim=1)(10*q_batch), axis=0)
            P = P/P.sum()
            Q = torch.ones(n_clusters)/n_clusters # Uniform distribution

            if args.cluster_balance == "kl":
                cluster_balance_loss = F.kl_div(P.log(), Q, reduction='batchmean')
            else:
                cluster_balance_loss = torch.linalg.norm(torch.sqrt(P) - torch.sqrt(Q))

            km_loss = F.kl_div(q_batch.log(), p_train[idx], reduction='batchmean')

            loss = reconstr_loss
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

        # print('Epoch: {:02d} | Epoch KM Loss: {:.3f} | Total Loss: {:.3f} | Classification Loss: {:.3f} |\
        # Cluster Balance Loss: {:.3f}'.format(epoch, epoch_km_loss, epoch_loss, epoch_class_loss, loss))
        # train_losses.append([np.round(epoch_loss.item(),3), np.round(epoch_class_loss.item(),3)])
    model = best_model

    q_train, z_train = model.encoder_forward(torch.FloatTensor(np.array(X_train)).to(args.device), output="latent")
    cluster_id_train = torch.argmax(q_train, axis=1)

    X_latents_data_loader = list(zip(z_train.to(args.device),q_train, y_train))

    train_loader_latents = torch.utils.data.DataLoader(X_latents_data_loader,
        batch_size=1024, shuffle=False)

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
    
        tune.report(loss=val_loss.detach().item(), accuracy=val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_auprc = val_auprc
            best_model = model
            best_val_loss = val_loss


    # Evaluate model on Test dataset
    model = best_model
    q_test, z_test = model.encoder_forward(torch.FloatTensor(X_test).to(args.device), output="latent")
    cluster_ids = torch.argmax(q_test, axis=1)

    test_loss = 0
    e_test_loss = 0
    local_sum_loss = 0

    test_preds_e = model.predict(torch.FloatTensor(X_test).to(args.device), attention=True)
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

    print('Loss Metrics - Test Loss {:.3f}, E-Test Loss {:.3f}, Local Sum Test Loss {:.3f}'.format(test_loss, e_test_loss, local_sum_loss))

    print('Clustering Metrics - Acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi),\
          ', ari {:.4f}, HTFD {:.3f}'.format(ari, e_test_HTFD))

    print('Classification Metrics - Test F1 {:.3f}, Test AUC {:.3f}, Test AUPRC {:.3f}, Test MIN_PSE {:.3f}'.format(test_f1,\
        test_auc, test_auprc, test_minpse), ', E-Test F1 {:.3f}, E-Test AUC {:.3f}, E-Test AUPRC {:.3f}, E-Test MIN_PSE {:.3f}'.format\
        (e_test_f1, e_test_auc, e_test_auprc, e_test_minpse))

    print("\n")



def main(num_samples=15, max_num_epochs=50, cpus_per_trial=12):
    config = {
        "lr_exp": tune.loguniform(1e-4, 1e-1),
        "lr_enc": tune.loguniform(1e-4, 1e-1),
        "beta": tune.uniform(1e-1, 1e+1),
        "gamma": tune.uniform(1e-1, 1e+1),
        "delta": tune.uniform(1e-1, 1e+1),
        "n_clusters": tune.choice([2,3,4,5,6,7,8,9,10])
        # "batch_size": tune.choice([2, 4, 8, 16])
        }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy"])
    
    result = tune.run(
        partial(train_expertnet, args=args),
        resources_per_trial={"cpu": 6},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # # best_checkpoint_dir = best_trial.checkpoint.value
    # # model_state, optimizer_state = torch.load(os.path.join(
    # #     best_checkpoint_dir, "checkpoint"))
    # # best_trained_model.load_state_dict(model_state)

    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=25, max_num_epochs=25, cpus_per_trial=12)

# enablePrint()

# print("[Avg]\tDataset\tk\tE-F1\tE-AUC\tE-AUPRC\tE-MINPSE\tE-ACC")
# print("\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(args.dataset, n_clusters,\
#     np.avg(e_f1_scores), np.avg(e_auc_scores), np.avg(e_auprc_scores), np.avg(e_minpse_scores), np.avg(e_acc_scores)))

# print("[Std]\tE-F1\tE-AUC\tE-AUPRC\tE-MINPSE\tE-ACC")
# print("\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format\
#     (np.std(e_f1_scores), np.std(e_auc_scores), np.std(e_auprc_scores), np.std(e_minpse_scores), np.std(e_acc_scores)))

# print('[Avg]\tTr-SIL\tHTFD\tWDFD')
# print("\t{:.3f}\t{:.3f}\t{:.3f}\n".format(np.avg(sil_scores),\
#     np.avg(HTFD_scores), np.avg(wdfd_scores)))

# print('[Std]\tTr-SIL\tHTFD\tWDFD')
# print("\t{:.3f}\t{:.3f}\t{:.3f}\n".format(np.std(sil_scores),\
#     np.std(HTFD_scores), np.std(wdfd_scores)))

# # print("F1\tAUC\tAUPRC\tACC")

# # print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format\
# #     (np.avg(f1_scores), np.avg(auc_scores), np.avg(auprc_scores), np.avg(acc_scores)))

# # print("Train Loss\tE-Train Loss\tTest Loss\tE-Test Loss")

# # print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format\
# #     (np.avg(train_losses), np.avg(e_train_losses),\
# #     np.avg(test_losses), np.avg(e_test_losses)))

# if args.cluster_analysis == "True":
#     WDFD_Cluster_Analysis(torch.Tensor(X_train), cluster_ids_train, column_names)
#     HTFD_Cluster_Analysis(torch.Tensor(X_train), cluster_ids_train, column_names)
#     HTFD_Single_Cluster_Analysis(X_train, y_train, cluster_ids_train, column_names)