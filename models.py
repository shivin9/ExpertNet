import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import torch
from utils import is_non_zero_file
from collections import OrderedDict

class AE(nn.Module):
    def __init__(self, layers):
        super(AE, self).__init__()

        self.encoder = OrderedDict()
        self.decoder = OrderedDict()
        n_layers = int(len(layers)/2)
        for i in range(n_layers-1):
            self.encoder.update(
                {"layer{}".format(i): nn.Linear(layers[i], layers[i+1]),
                'activation{}'.format(i): nn.ReLU(),
                })


        self.encoder = nn.Sequential(self.encoder)
        self.z_layer = nn.Linear(layers[n_layers-1], layers[n_layers])

        for i in range(n_layers, 2*n_layers-1):
            self.decoder.update(
                {"layer{}".format(i): nn.Linear(layers[i], layers[i+1]),
                'activation{}'.format(i): nn.ReLU(),
                })

        self.decoder = nn.Sequential(self.decoder)
        self.x_bar_layer = nn.Linear(layers[2*n_layers-1], layers[2*n_layers])


    def forward(self, x, output="decoded"):

        # encoder
        enc = self.encoder(x)
        z = self.z_layer(enc)
        if output == "latent":
            return z

        # decoder
        dec = self.decoder(z)
        x_bar = self.x_bar_layer(dec)

        return x_bar, z


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def source_distribution(z, cluster_layer, alpha=1):
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(z.unsqueeze(1) - cluster_layer, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q


def pretrain_ae(model, train_loader, args):
    '''
    pretrain autoencoder
    '''
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(50):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(args.device)

            optimizer.zero_grad()
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("Pretraining epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))


class NNClassifier(nn.Module):
    def __init__(self, args, input_dim, ae=None):
        super(NNClassifier, self).__init__()
        self.args = args
        self.n_classes = args.n_classes
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        # self.criterion = torch.nn.HingeEmbeddingLoss(reduction='mean')
        self.input_dim = args.input_dim
        self.alpha = args.alpha
        self.gamma = args.gamma

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, args.n_z),
        )

        self.decoder = nn.Sequential(
            nn.Linear(args.n_z, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, args.input_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(args.n_z, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, args.n_classes),
        )

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=args.lr)

    def forward(self, inputs):
        z = self.encoder(inputs)
        return self.classifier(z), self.decoder(z)

    def fit(self, X_batch, y_batch):
        self.optimizer.zero_grad()
        self.classifier.train()
        y_pred, x_bar = self.forward(X_batch)
        train_loss = self.criterion(y_pred, y_batch)
        reconstr_loss = F.mse_loss(x_bar, X_batch)
        total_loss = self.alpha*reconstr_loss + self.gamma*train_loss
        total_loss.backward()
        self.optimizer.step()
        return y_pred.detach().numpy(), train_loss.item()


class NNClassifierBase(nn.Module):
    def __init__(self, args, input_dim, layers):
        super(NNClassifierBase, self).__init__()
        self.args = args
        self.n_classes = args.n_classes
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        # self.criterion = torch.nn.HingeEmbeddingLoss(reduction='mean')
        self.device = args.device
        self.input_dim = args.input_dim
        self.alpha = args.alpha
        self.gamma = args.gamma
        n_layers = len(layers)

        self.classifier = OrderedDict()
        for i in range(n_layers-2):
            self.classifier.update(
                {"layer{}".format(i): nn.Linear(layers[i], layers[i+1]),
                'activation{}'.format(i): nn.ReLU(),
                })

        i = n_layers - 2
        self.classifier.update(
            {"layer{}".format(i): nn.Linear(layers[i], layers[i+1]),
            })

        self.classifier = nn.Sequential(self.classifier).to(self.device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=args.lr)

    def forward(self, inputs):
        return self.classifier(inputs)

    def fit(self, X_batch, y_batch):
        self.optimizer.zero_grad()
        self.classifier.train()
        y_pred = self.forward(X_batch)
        train_loss = self.criterion(y_pred, y_batch)
        total_loss = train_loss
        total_loss.backward()
        self.optimizer.step()
        return y_pred.detach().numpy(), train_loss.item()


class DeepCAC(nn.Module):
    def __init__(self,
                 ae_layers,
                 expert_layers,
                 args):
        super(DeepCAC, self).__init__()
        self.alpha = args.alpha
        self.pretrain_path = args.pretrain_path
        self.device = args.device
        self.n_clusters = args.n_clusters
        self.input_dim = args.input_dim
        self.n_z = args.n_z
        self.args = args
        ae_layers.append(self.input_dim)
        ae_layers = [self.input_dim] + ae_layers
        self.ae = AE(ae_layers)
        # cluster layer
        self.cluster_layer = torch.Tensor(self.n_clusters, self.n_z)
        self.class_cluster_layer = torch.Tensor(self.n_clusters, args.n_classes, self.n_z)

        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        torch.nn.init.xavier_normal_(self.class_cluster_layer.data)
        
        self.classifiers = []
        n_layers = int(len(expert_layers))
        for _ in range(self.n_clusters):
            classifier = OrderedDict()
            for i in range(n_layers-2):
                classifier.update(
                    {"layer{}".format(i): nn.Linear(expert_layers[i], expert_layers[i+1]),
                    'activation{}'.format(i): nn.ReLU(),
                    })

            i = n_layers-2
            classifier.update(
                {"layer{}".format(i): nn.Linear(expert_layers[i], expert_layers[i+1]),
                })


            classifier = nn.Sequential(classifier).to(self.device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
            self.classifiers.append([classifier, optimizer])
            

    def pretrain(self, train_loader, path=''):
        print(path)
        if not is_non_zero_file(path):
            path = ''
        if path == '':
            pretrain_ae(self.ae, train_loader, self.args)
        else:
            # load pretrain weights
            self.ae.load_state_dict(torch.load(self.pretrain_path))
            print('load pretrained ae from', path)


    def predict(self, X_test):
        z_test = self.forward(X_test)
        # cluster_ids = torch.argmax(q_test, axis=1)
        preds = torch.zeros((self.n_clusters, 2))
        for j in range(self.n_clusters):
            preds[j,:] = self.classifiers[cluster_ids[j]]
        return preds


    def forward(self, x, output="default"):
        x_bar, z = self.ae(x)

        if output == "latent":
            return z, x_bar

        elif output == "classifier":
            preds = torch.zeros((len(z), 2))
            for j in range(len(z)):
                preds[j,:] = self.classifiers[j](z)
            return preds
        
        else:
            return z


class ExpertNet(nn.Module):
    def __init__(self,
                 ae_layers,
                 expert_layers,
                 args):
        super(ExpertNet, self).__init__()
        self.alpha = args.alpha
        self.pretrain_path = args.pretrain_path
        self.device = args.device
        self.n_clusters = args.n_clusters
        self.input_dim = args.input_dim
        self.n_z = args.n_z
        self.args = args
        ae_layers.append(self.input_dim)
        ae_layers = [self.input_dim] + ae_layers
        self.ae = AE(ae_layers)
        # cluster layer
        self.cluster_layer = torch.Tensor(self.n_clusters, self.n_z)
        self.p_cluster_layer = torch.Tensor(self.n_clusters, self.n_z)
        self.n_cluster_layer = torch.Tensor(self.n_clusters, self.n_z)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        torch.nn.init.xavier_normal_(self.p_cluster_layer.data)
        torch.nn.init.xavier_normal_(self.n_cluster_layer.data)

        self.classifiers = []
        n_layers = int(len(expert_layers))
        for _ in range(self.n_clusters):
            classifier = OrderedDict()
            for i in range(n_layers-2):
                classifier.update(
                    {"layer{}".format(i): nn.Linear(expert_layers[i], expert_layers[i+1]),
                    'activation{}'.format(i): nn.ReLU(),
                    })

            i = n_layers - 2
            classifier.update(
                {"layer{}".format(i): nn.Linear(expert_layers[i], expert_layers[i+1]),
                })

            classifier = nn.Sequential(classifier).to(self.device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
            self.classifiers.append([classifier, optimizer])
            

    def pretrain(self, train_loader, path=''):
        print(path)
        if not is_non_zero_file(path):
            path = ''
        if path == '':
            pretrain_ae(self.ae, train_loader, self.args)
        else:
            # load pretrain weights
            self.ae.load_state_dict(torch.load(self.pretrain_path))
            print('load pretrained ae from', path)


    def predict(self, X_test):
        qs, z_test = self.forward(X_test)
        q_test = qs[0]
        cluster_ids = torch.argmax(q_test, axis=1)
        preds = torch.zeros((self.n_clusters, 2))
        for j in range(self.n_clusters):
            preds[j,:] = self.classifiers[cluster_ids[j]]
        return preds


    def forward(self, x, output="default"):
        x_bar, z = self.ae(x)
        # Cluster
        q   = source_distribution(z, self.cluster_layer, alpha=self.alpha)
        q_p = source_distribution(z, self.p_cluster_layer, alpha=self.alpha)
        q_n = source_distribution(z, self.n_cluster_layer, alpha=self.alpha)

        if output == "latent":
            return (q, q_p, q_n), z

        elif output == "classifier":
            preds = torch.zeros((len(z), 2))
            for j in range(len(z)):
                preds[j,:] = self.classifiers[j](z)
            return preds
        
        else:
            return z, x_bar, (q, q_p, q_n)


class DMNN(nn.Module):
    def __init__(self,
                 ae_layers,
                 expert_layers,
                 args):
        super(DMNN, self).__init__()
        self.alpha = args.alpha
        self.pretrain_path = args.pretrain_path
        self.device = args.device
        self.n_clusters = args.n_clusters
        self.input_dim = args.input_dim
        self.n_z = args.n_z
        self.args = args
        self.n_classes = args.n_classes
        ae_layers.append(self.input_dim)
        ae_layers = [self.input_dim] + ae_layers
        self.ae = AE(ae_layers)

        # Gating layer
        self.gate = nn.Sequential(
            Linear(self.n_z, self.n_clusters),
            nn.Softmax(dim=1)
            ).to(self.device)

        # cluster layer
        self.cluster_layer = torch.Tensor(self.n_clusters, self.n_z)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        n_layers = int(len(expert_layers))
        for _ in range(self.n_clusters):
            classifier = OrderedDict()
            for i in range(n_layers-2):
                classifier.update(
                    {"layer{}".format(i): nn.Linear(expert_layers[i], expert_layers[i+1]),
                    'activation{}'.format(i): nn.ReLU(),
                    })

            i = n_layers - 2
            classifier.update(
                {"layer{}".format(i): nn.Linear(expert_layers[i], expert_layers[i+1]),
                })

            classifier = nn.Sequential(classifier).to(self.device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
            self.classifiers.append([classifier, optimizer])            

    def pretrain(self, train_loader, path=''):
        print(path)
        if not is_non_zero_file(path):
            path = ''
        if path == '':
            pretrain_ae(self.ae, train_loader, self.args)
        else:
            # load pretrain weights
            self.ae.load_state_dict(torch.load(self.pretrain_path))
            print('load pretrained ae from', path)


    def predict(self, X_test):
        qs, z_test = self.forward(X_test)
        q_test = qs[0]
        cluster_ids = torch.argmax(q_test, axis=1)
        preds = torch.zeros((self.n_clusters, self.n_classes))
        for j in range(self.n_clusters):
            preds[j,:] = self.classifiers[cluster_ids[j]]
        return preds


    def forward(self, x, output="default"):
        x_bar, z = self.ae(x)
        g = self.gate(z)

        if output == "latent":
            return z
        
        else:
            return z, x_bar, g