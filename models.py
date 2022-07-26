import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import torch
from utils import is_non_zero_file
from collections import OrderedDict
from ts_utils import batch_iter, pad_sents

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
    optimizer = Adam(model.parameters(), lr=args.lr_enc)
    for epoch in range(200):
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


def pretrain_ts_ae(model, train, args):
    optimizer = Adam(model.parameters(), lr=args.lr_enc)
    X_train, y_train, X_train_len = train
    pad_token = np.zeros(args.input_dim)
    for epoch in range(200):
        total_loss = 0.
        for batch_idx, (idx_batch, x_batch, y_batch, batch_lens) in\
            enumerate(batch_iter(X_train, y_train, X_train_len, args.batch_size, shuffle=True)):
            
            x_batch = torch.tensor(pad_sents(x_batch, pad_token, args.n_features, args.end_t), dtype=torch.float32).to(args.device)
            x_batch = torch.nan_to_num(x_batch)
            x_batch = x_batch.to(args.device)
            optimizer.zero_grad()
            x_bar, _, _ = model.ae(x_batch)
            # print("x_bar", x_bar)
            # print("x_batch", x_batch)
            loss = F.mse_loss(x_bar, x_batch)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("Pretraining epoch {} loss = {:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.ae.state_dict(), args.pretrain_path)
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


    def forward(self, inputs):
        input_bar, z = self.ae(inputs)
        last_tensor = self.classifier(z)
        return input_bar, last_tensor, self.last_layer(last_tensor)


    def fit(self, X_batch, y_batch):
        self.optimizer.zero_grad()
        self.classifier.train()
        x_bar, _, y_pred = self.forward(X_batch)
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
        return None, None, self.classifier(inputs)

    def fit(self, X_batch, y_batch):
        self.optimizer.zero_grad()
        self.classifier.train()
        _, _, y_pred = self.forward(X_batch)
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
                 ae_layers,
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
        # self.bn = nn.BatchNorm1d(self.hidden_dim)

        # append input_dim at the end
        ae_layers.append(self.input_dim)
        ae_layers = [self.input_dim] + ae_layers
        self.ae = AE(ae_layers)
        # cluster layer
        self.cluster_layer = torch.Tensor(self.n_clusters, self.n_z)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        # nn.init.xavier_uniform_(self.ae.encoder.weight)
        # nn.init.xavier_uniform_(self.ae.decoder.weight)

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
        print(path)
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
                cluster_id = np.where(cluster_ids == j)[0]
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
                # print(cluster_loss, X_cluster)
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


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim=1, dropout_prob=0.7):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # GRU layers
        self.gru_enc1 = nn.GRU(
            input_dim, hidden_dim[0], layer_dim, batch_first=True, dropout=dropout_prob
        )

        self.gru_enc2 = nn.GRU(
            hidden_dim[0], hidden_dim[1], layer_dim, batch_first=True, dropout=dropout_prob
        )

        self.gru_enc3 = nn.GRU(
            hidden_dim[1], hidden_dim[2], layer_dim, batch_first=True, dropout=dropout_prob
        )

        self.gru_dec1 = nn.GRU(
            hidden_dim[2], hidden_dim[1], layer_dim, batch_first=True, dropout=dropout_prob
        )

        self.gru_dec2 = nn.GRU(
            hidden_dim[1], hidden_dim[0], layer_dim, batch_first=True, dropout=dropout_prob
        )

        self.gru_dec3 = nn.GRU(
            hidden_dim[0], hidden_dim[0], layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim[0], input_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim[0]).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        enc_out1, h_enc1 = self.gru_enc1(x, h0.detach())
        enc_out2, h_enc2 = self.gru_enc2(enc_out1)
        enc_out3, h_enc3 = self.gru_enc3(enc_out2)

        dec_out1, h_dec1 = self.gru_dec1(enc_out3)
        dec_out2, h_dec2 = self.gru_dec2(dec_out1)
        dec_out3, h_dec3 = self.gru_dec3(dec_out2)

        # print(dec_out.shape)
        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        hidden = enc_out3[:, -1, :].squeeze()

        batch_size = dec_out3.shape[0]
        seq_len = dec_out3.shape[1]
        n_features = dec_out3.shape[2]
        # Convert the final state to our desired output shape (batch_size, output_dim)
        x_bar = self.fc(dec_out3.reshape(batch_size*seq_len, n_features))
        # print(x_bar.shape)
        return x_bar.reshape(batch_size, seq_len, self.input_dim), hidden, hidden


# class Encoder(nn.Module):
#     def __init__(self, batch_size, seq_len, n_features, embedding_dim=64):
#         super(Encoder, self).__init__()
#         self.batch_size, self.seq_len, self.n_features = batch_size, seq_len, n_features
#         self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
#         self.rnn1 = nn.LSTM(
#           input_size=n_features,
#           hidden_size=self.hidden_dim,
#           num_layers=1,
#           batch_first=True
#         )
#         self.rnn2 = nn.LSTM(
#           input_size=self.hidden_dim,
#           hidden_size=embedding_dim,
#           num_layers=1,
#           batch_first=True
#         )

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = x.reshape((batch_size, self.seq_len, self.n_features))
#         x, (_, _) = self.rnn1(x)
#         x, (hidden_n, _) = self.rnn2(x)
#         return hidden_n.reshape((batch_size, self.embedding_dim))


# class Decoder(nn.Module):
#     def __init__(self, batch_size, seq_len, embedding_dim, input_dim=64):
#         super(Decoder, self).__init__()
#         self.batch_size, self.seq_len, self.embedding_dim, self.input_dim = batch_size, seq_len, embedding_dim, input_dim
#         self.hidden_dim = 2 * embedding_dim
#         self.rnn1 = nn.LSTM(
#           input_size=embedding_dim,
#           hidden_size=embedding_dim,
#           num_layers=1,
#           batch_first=True
#         )
#         self.rnn2 = nn.LSTM(
#           input_size=embedding_dim,
#           hidden_size=self.hidden_dim,
#           num_layers=1,
#           batch_first=True
#         )
#         self.output_layer = nn.Linear(self.hidden_dim, input_dim)

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = x.repeat(self.seq_len, 1)
#         x = x.reshape((batch_size, self.seq_len, self.embedding_dim))
#         x, (hidden_n, cell_n) = self.rnn1(x)
#         x, (hidden_n, cell_n) = self.rnn2(x)
#         x = x.reshape((batch_size*self.seq_len, self.hidden_dim))
#         out = self.output_layer(x)
#         return out.reshape((batch_size, self.seq_len, self.input_dim))


# class RecurrentAutoencoder(nn.Module):
#     def __init__(self, batch_size, seq_len, n_features, device, embedding_dim=64):
#         super(RecurrentAutoencoder, self).__init__()
#         self.encoder = Encoder(batch_size, seq_len, n_features, embedding_dim).to(device)
#         self.decoder = Decoder(batch_size, seq_len, embedding_dim, n_features).to(device)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


class ExpertNet_GRU(nn.Module):
    def __init__(self,
                 expert_layers,
                 lr_enc,
                 lr_exp,
                 args):
        super(ExpertNet_GRU, self).__init__()
        self.alpha = args.alpha
        self.pretrain_path = args.pretrain_path
        self.device = args.device
        self.n_clusters = args.n_clusters
        self.n_classes = args.n_classes
        self.n_features = args.n_features
        self.n_z = args.n_z
        self.args = args
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.args = args
        self.lr_exp = lr_exp
        self.lr_enc = lr_enc
        self.batch_size = args.batch_size
        self.seq_len = args.end_t

        self.hidden_dim = [128, 64, self.n_z]

        # append input_dim at the end
        # self.ae = GRUModel(self.n_features, self.n_z)
        self.ae = GRUModel(self.n_features, self.hidden_dim)
        # self.ae = RecurrentAutoencoder(self.batch_size, self.seq_len, self.n_features, self.device, self.n_z)

        # cluster layer
        self.cluster_layer = torch.Tensor(self.n_clusters, self.n_z)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

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
            

    def pretrain(self, train, path=''):
        print(self.ae)

        if not is_non_zero_file(path):
            path = ''
        if path == '':
            print("Pretraining AE")
            pretrain_ts_ae(self, train, self.args)
        else:
            # load pretrain weights
            self.ae.load_state_dict(torch.load(self.pretrain_path))
            # pretrain_ts_ae(self, train, self.args)
            print('Loading Pretrained AE from', path)


    def encoder_forward(self, x, output="default"):
        x_bar, _, z = self.ae(x)
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


    def predict(self, X=None, z=None, q=None, attention=True):
        if z != None and q != None:
            z = z
            q = q
        else:
            z, _, q = self.encoder_forward(X)

        preds = torch.zeros((len(z), self.n_classes))

        cluster_ids = torch.argmax(q, axis=1)
        X_cluster = z
        total_loss = 0
        for j in range(self.n_clusters):
            if attention == True:
                # Weighted predictions
                cluster_id = np.where(cluster_ids == j)[0]
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
            z, out, q = self.encoder_forward(X)

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
                    y_pred_cluster = classifier_k(X_cluster) # Backprop the error to encoder
                else:
                    y_pred_cluster = classifier_k(X_cluster.detach()) # Do not backprop the error to encoder

                cluster_loss = torch.sum(self.criterion(y_pred_cluster, y_cluster))

            else:
                X_cluster = z
                if backprop_enc == True:
                    y_pred_cluster = classifier_k(X_cluster)
                    cluster_loss = torch.sum(q[:,k]*self.criterion(y_pred_cluster, y_cluster))
                else:
                    y_pred_cluster = classifier_k(X_cluster.detach())
                    cluster_loss = torch.sum(q.detach()[:,k]*self.criterion(y_pred_cluster, y_cluster))

                # cluster_loss = Variable(cluster_loss.data, requires_grad=True)

            if backprop_local == True:
                optimizer_k.zero_grad()
                cluster_loss.backward(retain_graph=True)
                optimizer_k.step()
            total_loss += cluster_loss

        return q, total_loss