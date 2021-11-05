import torch
from torch import nn
from utils import utils
from collections import OrderedDict


class GAE(object):
    def __init__(self, n, d, x_dim, seed=8, num_encoder_layer=1, num_decoder_layer=1, hidden_dim=5, latent_dim=1, l1_graph_penalty=0, device='cuda'):
        super(GAE, self).__init__()
        self.n = n
        self.d = d
        self.x_dim = x_dim
        self.seed = seed
        self.num_encoder_layer = num_encoder_layer
        self.num_decoder_layer = num_decoder_layer
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.l1_graph_penalty = l1_graph_penalty

        self.device = device

        self.encoder = self._define_net(encoder=1).to(self.device)

        self.decoder = self._define_net(encoder=0).to(self.device)

        self._build()


    def _define_net(self, encoder=1):
        if encoder:
            if self.num_encoder_layer == 1:
                return nn.Linear(self.x_dim, self.latent_dim)
            elif self.num_encoder_layer < 1:
                print('Please set a encoder layer number larger than 0')
            else:
                layers = OrderedDict()
                layers['0'] = nn.Linear(self.x_dim, self.hidden_dim)
                layers['0_a'] = nn.LeakyReLU(negative_slope=0.05)
                for i in range(1, self.num_encoder_layer-1):
                    layers[str(i)] = nn.Linear(self.hidden_dim, self.hidden_dim)
                    layers[str(i)+'_a'] = nn.LeakyReLU(negative_slope=0.05)
                layers[str(self.num_encoder_layer-1)] = nn.Linear(self.hidden_dim, self.latent_dim)
                return nn.Sequential(layers)
        else:
            if self.num_decoder_layer == 1:
                return nn.Linear(self.latent_dim, self.x_dim)
            elif self.num_decoder_layer < 1:
                print('Please set a decoder layer number larger than 0')
            else:
                layers = OrderedDict()
                layers['0'] = nn.Linear(self.latent_dim, self.hidden_dim)
                layers['0_a'] = nn.LeakyReLU(negative_slope=0.05)
                for i in range(1, self.num_decoder_layer-1):
                    layers[str(i)] = nn.Linear(self.hidden_dim, self.hidden_dim)
                    layers[str(i)+'_a'] = nn.LeakyReLU(negative_slope=0.05)
                layers[str(self.num_decoder_layer-1)] = nn.Linear(self.hidden_dim, self.x_dim)
                return nn.Sequential(layers)

    def _build(self):
        utils.set_seed(self.seed)
        w = torch.empty(self.d, self.d)
        nn.init.uniform_(w, a=-0.1, b=0.1)
        mask = torch.eye(self.d, self.d)
        self.w_prime = w.masked_fill_(mask, 0)


    def _get_mse_loss(self, X, W_prime):
        X_prime = self.encoder.forward(X)
        X_prime = torch.einsum('ijk,jl->ilk', X_prime, W_prime)
        X_prime = self.decoder.forward(X_prime)
        return nn.MSELoss(X, X_prime)
