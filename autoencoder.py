import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.output_dim = self.input_dim
        self.hidden_dims = args.hidden_dims
        self.hidden_dims.append(args.latent_dim)
        self.dims_list = (args.hidden_dims + 
                          args.hidden_dims[:-1][::-1])  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.n_clusters = args.n_clusters
        self.n_classes = args.n_classes
        
        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == args.latent_dim
        
        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {'linear0': nn.Linear(self.input_dim, hidden_dim),
                     'activation0': nn.ReLU()
                    })
            else:
                layers.update(
                    {'linear{}'.format(idx): nn.Linear(
                        self.hidden_dims[idx-1], hidden_dim),
                     'activation{}'.format(idx): nn.ReLU(),
                     'bn{}'.format(idx): nn.BatchNorm1d(self.hidden_dims[idx])
                    })
        self.encoder = nn.Sequential(layers)
        
        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {'linear{}'.format(idx): nn.Linear(
                        hidden_dim, self.output_dim),
                    })
            else:
                layers.update(
                    {'linear{}'.format(idx): nn.Linear(
                        hidden_dim, tmp_hidden_dims[idx+1]),
                     'activation{}'.format(idx): nn.ReLU(),
                     'bn{}'.format(idx): nn.BatchNorm1d(tmp_hidden_dims[idx+1])
                    })
        self.decoder = nn.Sequential(layers)

        # Classification Head
        self.classifiers = []
        for _ in range(self.n_clusters):
            self.classifiers.append(
                nn.Sequential(
                    nn.Linear(20, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, self.n_classes)))


    def __repr__(self):
        repr_str = '[Structure]: {}-'.format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
                repr_str += '{}-'.format(dim)
        repr_str += str(self.output_dim) + '\n'
        repr_str += '[n_layers]: {}'.format(self.n_layers) + '\n'
        repr_str += '[n_clusters]: {}'.format(self.n_clusters) + '\n'
        repr_str += '[n_classes]: {}'.format(self.n_classes) + '\n'
        repr_str += '[input_dims]: {}'.format(self.input_dim)
        return repr_str
    
    def __str__(self):
        return self.__repr__()
    
    def forward(self, X, latent=False, classifier_idx=0):
        body_output = self.encoder(X)
        if latent:
            return body_output
        probs = self.classifiers[classifier_idx](body_output)
        return self.decoder(body_output), probs