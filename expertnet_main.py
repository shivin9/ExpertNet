from expertnet_base import *

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default= 'creditcard')
parser.add_argument('--input_dim', default= '-1')
parser.add_argument('--n_features', default= '-1')
parser.add_argument('--target', default= -1, type=int)
parser.add_argument('--data_ratio', default= 1, type=float)
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
parser.add_argument("--load_ae", default= False, type=bool)
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
parser.add_argument('--ae_type', default= 'dae')
parser.add_argument('--n_channels', default= 1, type=int)

# Utility parameters
parser.add_argument('--device', default= 'cpu')
parser.add_argument('--verbose', default= 'False')
parser.add_argument('--plot', default= 'False')
parser.add_argument('--expt', default= 'ExpertNet')
parser.add_argument('--cluster_analysis', default= 'False')
parser.add_argument('--log_interval', default= 10, type=int)
parser.add_argument('--pretrain_path', default= '/Users/shivin/Document/NUS/Research/CAC/CAC_DL/ExpertNet/pretrained_model/EN')
# parser.add_argument('--pretrain_path', default= '/home/shivin/CAC_code/data')

def main():
    # Trained model and modified training data and labels
    args = parameters(params)
    args.target = 3
    args.n_clusters = 3
    model, train, val, test, column_names = run_ExpertNet(args)

    args = parameters(params)

    # Run Fashion MNIST
    for k in range(10):
        args.ae_type = 'cnn'
        args.dataset = 'FashionMNIST'
        args.target = -1
        args.n_classes = 10
        args.verbose = False
        args.n_clusters = k
        model, train, val, test, column_names = run_ExpertNet(args)

    args.ae_type = 'cnn'
    args.dataset = 'FashionMNIST'
    args.target = -1
    args.n_classes = 10
    args.verbose = False
    args.n_clusters = 3
    data_ratios = [1, 0.9, 0.75, 0.5, 0.4, 0.25, 0.1, 0.01]

    # Run Fashion MNIST
    for dr in data_ratios:
        args.data_ratio = dr
        model, train, val, test, column_names = run_ExpertNet(args)

if __name__ == "__main__":
    main()