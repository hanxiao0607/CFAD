import torch
from torch import nn
from utils import utils
from collections import OrderedDict
from itertools import chain
from torch.autograd import Variable



class GAE(object):
    def __init__(self, n, d, x_dim, seed=8, num_encoder_layer=1, num_decoder_layer=1, hidden_dim=5, latent_dim=1, l1_graph_penalty=0, lr=0.001, device='cpu'):
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

        self.lr = lr

        self.device = device

        self.encoder = self._define_net(encoder=1).to(self.device)

        self.decoder = self._define_net(encoder=0).to(self.device)

        self.x = torch.zeros((self.n, self.d, self.x_dim))

        self.MSE = nn.MSELoss()

        self._build()

        self.all_params = chain(self.encoder.parameters(), self.W_prime, self.decoder.parameters())
        self.train_op = torch.optim.Adam(self.all_params, lr=self.lr)


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
        mask = (mask>0)
        self.W_prime = torch.clone(w.masked_fill_(mask, 0.0))

        self.mse_loss = self._get_mse_loss(self.x, self.W_prime)
        self.h = self._get_h()

        self.alpha = 0.0
        self.rho = 0.0
        self.lr = 0.0

        self.loss = self._update_loss()

    def _get_h(self):
        return torch.trace(torch.exp(self.W_prime*self.W_prime)) - self.d

    def _update_loss(self):
        return 0.5 / self.n * self.mse_loss + self.l1_graph_penalty * torch.norm(self.W_prime, p=1) \
                    + self.alpha * self.h + 0.5 * self.rho * self.h * self.h


    def _get_mse_loss(self, X, W_prime):
        X_flattened = torch.reshape(X, (-1, self.x_dim))
        X_encoded_flattened = self.encoder.forward(X_flattened)
        X_encoded = torch.reshape(X_encoded_flattened, (self.n, self.d, self.latent_dim))
        X_encoded_prime = torch.einsum('ijk,jl->ilk', X_encoded, W_prime)
        X_encoded_prime_flattened = torch.reshape(X_encoded_prime, (-1, self.latent_dim))
        X_decoded_prime = self.decoder.forward(X_encoded_prime_flattened)
        X_decoded = torch.reshape(X_decoded_prime, (self.n, self.d, self.x_dim))
        return self.MSE(X, X_decoded)

    def _train(self, X, rho, alpha, lr):
        self.rho = rho
        self.alpha = alpha
        self.lr = lr
        self.X = X
        self.encoder.train()
        self.decoder.train()
        self.train_op.zero_grad()
        samples = torch.FloatTensor(self.X).to(self.device)
        self.mse_loss = self._get_mse_loss(samples, self.W_prime)
        loss = self._update_loss()
        self.loss = loss
        loss.backward()

        self.train_op.step()
        return self.loss, self.mse_loss, self._get_h(), self.W_prime


if __name__ == '__main__':
    n, d = 3000, 20

    model = GAE(n, d, x_dim=1)

    print('model.train_op: {}'.format(model.train_op))
    print('model.loss: {}'.format(model.loss))
    print('model.mse_loss: {}'.format(model.mse_loss))
    print('model.h: {}'.format(model.h))
    print('model.W_prime: {}'.format(model.W_prime))
    print('model.X: {}'.format(model.x))
    print('model.rho: {}'.format(model.rho))
    print('model.alpha: {}'.format(model.alpha))
    print('model.lr: {}'.format(model.lr))

    print('done')
