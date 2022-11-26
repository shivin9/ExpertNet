import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import torch
from utils import is_non_zero_file
from collections import OrderedDict
import numpy as np

class DAE(nn.Module):
    def __init__(self, layers):
        super(DAE, self).__init__()

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


# Define CNN classifier
class CNN_Classifier(nn.Module):
    def __init__(self, args, layers):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(args.n_channels, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.classifier = OrderedDict()
        layers[0] = 3 * 3 * 32
        n_layers = int(len(layers))
        for i in range(n_layers-1):
            self.classifier.update(
                {"layer{}".format(i): nn.Linear(layers[i], layers[i+1]),
                'activation{}'.format(i): nn.ReLU(),
                })

        self.classifier = nn.Sequential(self.classifier)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr_enc)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        x = self.encoder_cnn(x)
        x_hat = self.flatten(x)
        y_hat = self.classifier(x_hat)
        return x_hat, y_hat
    
    def fit(self, X_batch, y_batch):
        self.optimizer.zero_grad()
        # forward + backward + optimize
        input_hat, y_pred = self.forward(X_batch)
        train_loss = self.criterion(y_pred, y_batch)
        train_loss.backward()
        self.optimizer.step()
        return y_pred.detach().numpy(), train_loss.item()


# Define Autoencoder
class CNN_Encoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim, n_channels=1):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class CNN_Decoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim, n_channels=1):
        super().__init__()
        ### Convolutional section
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, n_channels, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class CNN_AE(nn.Module):
    def __init__(self, args, fc2_input_dim):
        super().__init__()
        ### Convolutional section
        self.encoder = CNN_Encoder(args.n_z, fc2_input_dim, args.n_channels)
        self.decoder = CNN_Decoder(args.n_z, fc2_input_dim, args.n_channels)
    
    def forward(self, x, output="decoded"):
        # encoder
        enc = self.encoder(x)
        if output == "latent":
            return enc

        # decoder
        x_bar = self.decoder(enc)
        return x_bar, enc


class CIFAR_AE(nn.Module):
    def __init__(self, args, fc2_input_dim):
        super(CIFAR_AE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            # nn.ReLU(),
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(4 * 4 * 48, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, args.n_z)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(48, 4, 4))

        self.decoder_lin = nn.Sequential(
            nn.Linear(args.n_z, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, 4 * 4 * 48),
            nn.ReLU(True)
        )

        self.decoder_conv = nn.Sequential(
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
        )

    def forward(self, x, output="decoded"):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        encoded_lin = self.encoder_lin(x)

        if output == "latent":
            return encoded_lin

        x = self.decoder_lin(encoded_lin)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        decoded = torch.sigmoid(x)

        return decoded, encoded_lin


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
    optimizer = Adam(model.parameters(), lr=args.lr_enc)
    for epoch in range(args.pre_epoch):
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
    def __init__(self, ae_layers, expert_layers, args):
        super(NNClassifier, self).__init__()
        self.args = args
        self.n_classes = args.n_classes
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        # self.criterion = torch.nn.HingeEmbeddingLoss(reduction='mean')
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.input_dim = args.input_dim
        self.n_z = args.n_z
        self.device = args.device
        self.args = args
        self.pretrain_path = args.pretrain_path

        # append input_dim at the end
        ae_layers.append(self.input_dim)
        ae_layers = [self.input_dim] + ae_layers
        self.ae = AE(ae_layers)

        n_layers = int(len(expert_layers))
        classifier = OrderedDict()
        for i in range(n_layers-2):
            classifier.update(
                {"layer{}".format(i): nn.Linear(expert_layers[i], expert_layers[i+1]),
                'activation{}'.format(i): nn.ReLU(),
                })

        i = n_layers - 2
        self.last_layer = nn.Linear(expert_layers[i], expert_layers[i+1])
        # classifier.update(
        #     {"layer{}".format(i): nn.Linear(expert_layers[i], expert_layers[i+1]),
        #     })

        self.classifier = nn.Sequential(classifier).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr_enc)


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


    def forward(self, inputs, output='default'):
        input_bar, z = self.ae(inputs)
        last_tensor = self.classifier(z)
        if output == 'default':
            return last_tensor, self.last_layer(last_tensor)
        else:
            return input_bar, last_tensor, self.last_layer(last_tensor)


    def fit(self, X_batch, y_batch):
        self.optimizer.zero_grad()
        self.classifier.train()
        x_bar, _, y_pred = self.forward(X_batch, output='reconstructed')
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
                # "bn{}".format(i): nn.BatchNorm1d(layers[i+1]),
                'activation{}'.format(i): nn.ReLU(),
                })

        i = n_layers - 2
        self.classifier.update(
            {"layer{}".format(i): nn.Linear(layers[i], layers[i+1]),
            # "bn{}".format(i): nn.BatchNorm1d(layers[i+1]),
            # "dropout{}".format(i): nn.Dropout(0.7)
            })

        self.classifier = nn.Sequential(self.classifier).to(self.device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=args.lr_enc)

    def forward(self, inputs):
        return None, self.classifier(inputs)

    def fit(self, X_batch, y_batch):
        self.optimizer.zero_grad()
        self.classifier.train()
        _, y_pred = self.forward(X_batch)
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

        # append input_dim at the end
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
            optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr_enc)
            self.classifiers.append([classifier, optimizer])
            
        # self.main = nn.Linear(expert_layers[i+1], args.n_classes)

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


    def forward(self, x, output="default"):
        x_bar, z = self.ae(x)

        if output == "latent":
            return z, x_bar

        elif output == "classifier":
            preds = torch.zeros((len(z), self.n_classes))
            for j in range(len(z)):
                preds[j,:] = self.classifiers[j](z)
            return preds
        
        else:
            return z


class ExpertNet(nn.Module):
    def __init__(self,
                 DAE_model,
                 expert_layers,
                 lr_enc,
                 lr_exp,
                 args):
        super(ExpertNet, self).__init__()
        self.pretrain_path = args.pretrain_path
        self.device = args.device
        self.n_clusters = args.n_clusters
        self.n_classes = args.n_classes
        self.input_dim = args.input_dim
        self.n_z = args.n_z
        self.args = args
        self.lr_exp = lr_exp
        self.lr_enc = lr_enc
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.alpha = 1
        self.ae = DAE_model
        # self.bn = nn.BatchNorm1d(self.hidden_dim)

        # cluster layer
        self.cluster_layer = torch.Tensor(self.n_clusters, self.n_z)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        # torch.nn.init.xavier_uniform_(self.ae.encoder.weight)
        # torch.nn.init.xavier_uniform_(self.ae.decoder.weight)

        self.classifiers = []
        n_layers = int(len(expert_layers))
        for _ in range(self.n_clusters):
            classifier = OrderedDict()
            for i in range(n_layers-2):
                classifier.update(
                    {"layer{}".format(i): nn.Linear(expert_layers[i], expert_layers[i+1]),
                    # "bn{}".format(i): nn.BatchNorm1d(expert_layers[i+1]),
                    'activation{}'.format(i): nn.ReLU(),
                    # 'dropout{}'.format(i): nn.dropout(0.7)
                    })

            i = n_layers - 2
            classifier.update(
                {"layer{}".format(i): nn.Linear(expert_layers[i], expert_layers[i+1]),
                })

            classifier = nn.Sequential(classifier).to(self.device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=self.lr_exp)
            self.classifiers.append([classifier, optimizer])
            # nn.init.xavier_uniform_(classifier.weight)
            
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.lr_enc)


    def pretrain(self, train_loader, path=''):
        if not is_non_zero_file(path):
            path = ''
        if path == '':
            pretrain_ae(self.ae, train_loader, self.args)
        else:
            # load pretrain weights
            self.ae.load_state_dict(torch.load(self.pretrain_path))
            print('load pretrained ae from', path)

     
    def predict(self, X, attention=True):
        z, _, q = self.encoder_forward(X)
        cluster_ids = torch.argmax(q, axis=1)
        preds = torch.zeros((len(X), self.n_classes))
        X_cluster = z
        total_loss = 0
        for j in range(self.n_clusters):
            if attention == True:
                # Weighted predictions
                X_cluster = z
                cluster_preds = self.classifiers[j][0](X_cluster)
                for c in range(self.n_classes):
                    preds[:,c] += q[:,j]*cluster_preds[:,c]

            else:
                cluster_id = np.where(cluster_ids == j)[0]
                X_cluster = z[cluster_id]
                cluster_test_preds = self.classifiers[j][0](X_cluster)
                preds[cluster_id,:] = cluster_test_preds
    
        return preds


    def expert_forward(self, X, y, z=None, q=None, backprop_enc=False, backprop_local=False, attention=True):
        if z == None and q == None:
            z, _, q = self.encoder_forward(X)

        cluster_ids = torch.argmax(q, axis=1)
        y_cluster = y
        total_loss = 0

        for k in range(self.n_clusters):
            classifier_k, optimizer_k = self.classifiers[k]

            if attention == False:
                cluster_id = np.where(cluster_ids == k)[0]
                X_cluster = z[cluster_id]
                y_cluster = y[cluster_id]
                if backprop_enc == True:
                    y_pred_cluster = classifier_k(X_cluster.detach()) # Do not backprop the error to encoder
                else:
                    y_pred_cluster = classifier_k(X_cluster) # Do not backprop the error to encoder

                cluster_loss = torch.sum(self.criterion(y_pred_cluster, y_cluster))

            else:
                # classifier_labels = np.argmax(q.detach().cpu().numpy(), axis=1)
                # for j in range(len(q)):
                #     classifier_labels[j] = np.random.choice(range(self.n_clusters), p = q[j].detach().numpy())

                # for k in range(self.n_clusters):
                #     idx_cluster = np.where(classifier_labels == k)[0]
                #     X_cluster = z[idx_cluster]
                #     y_cluster = y[idx_cluster]

                #     y_pred_cluster = classifier_k(X_cluster.detach())
                #     cluster_loss = torch.mean(self.criterion(y_pred_cluster, y_cluster))

                X_cluster = z
                if backprop_enc == True:
                    y_pred_cluster = classifier_k(X_cluster)
                    cluster_loss = torch.sum(q[:,k]*self.criterion(y_pred_cluster, y_cluster))
                else:
                    y_pred_cluster = classifier_k(X_cluster.detach())
                    cluster_loss = torch.sum(q.detach()[:,k]*self.criterion(y_pred_cluster, y_cluster))

            if backprop_local == True:
                optimizer_k.zero_grad()
                cluster_loss.backward(retain_graph=True)
                optimizer_k.step()
            total_loss += cluster_loss

        return q, total_loss


    def encoder_forward(self, x, output="default"):
        x_bar, z = self.ae(x)
        q = source_distribution(z, self.cluster_layer, alpha=self.alpha)

        if output == "latent":
            return q, z

        elif output == "classifier":
            preds = torch.zeros((len(z), self.n_classes))
            for j in range(len(z)):
                preds[j,:] = self.classifiers[j](z)
            return preds
        
        else:
            return z, x_bar, q


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
        self.classifiers = []

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
            optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr_enc)
            self.classifiers.append([classifier, optimizer])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr_enc)


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