import torch
from torch import nn
from collections import OrderedDict


class Net(nn.Module):
    def __init__(self, n, d, x_dim, num_encoder_layer=1, num_decoder_layer=1, hidden_dim=5, latent_dim=1,
                 device='cuda:0', seed=0):
        super(Net, self).__init__()

        self.seed = seed

        self.n = n
        self.d = d
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_encoder_layer = num_encoder_layer
        self.num_decoder_layer = num_decoder_layer
        self.device = device

        self.encoder = self._define_net(encoder=1)
        self.decoder = self._define_net(encoder=0)

        self._get_W_prime()

    def _get_W_prime(self):
        w = torch.empty(self.d, self.d).uniform_(-0.1, 0.1)
        w.fill_diagonal_(0.)

        self.W_prime = torch.nn.Parameter(w)
        self.W_prime.required_grad = True

    def forward(self, X):
        X_flattened = torch.reshape(X, (-1, self.x_dim))
        X_encoded_flattened = self.encoder.forward(X_flattened)
        X_encoded = torch.reshape(X_encoded_flattened, (self.n, self.d, self.latent_dim))
        X_encoded_prime = torch.einsum('ijk,jl->ilk', X_encoded, self.W_prime)
        X_encoded_prime_flattened = torch.reshape(X_encoded_prime, (-1, self.latent_dim))
        X_decoded_prime = self.decoder.forward(X_encoded_prime_flattened)
        X_decoded = torch.reshape(X_decoded_prime, (self.n, self.d, self.x_dim))
        return X_decoded

    def encode(self, X):
        X_flattened = torch.reshape(X, (-1, self.x_dim))
        X_encoded_flattened = self.encoder.forward(X_flattened)
        X_encoded = torch.reshape(X_encoded_flattened, (1, self.d, self.latent_dim))
        return X_encoded

    def einsum(self, X, W_prime):
        X_encoded_prime = torch.einsum('ijk,jl->ilk', X, W_prime)
        return X_encoded_prime

    def decode(self, X):
        X_encoded_prime_flattened = torch.reshape(X, (-1, self.latent_dim))
        X_decoded_prime = self.decoder.forward(X_encoded_prime_flattened)
        X_decoded = torch.reshape(X_decoded_prime, (1, self.d, self.x_dim))
        return X_decoded

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
                for i in range(1, self.num_encoder_layer):
                    layers[str(i)] = nn.Linear(self.hidden_dim, self.hidden_dim)
                    layers[str(i) + '_a'] = nn.LeakyReLU(negative_slope=0.05)
                layers[str(self.num_encoder_layer)] = nn.Linear(self.hidden_dim, self.latent_dim)
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
                for i in range(1, self.num_decoder_layer):
                    layers[str(i)] = nn.Linear(self.hidden_dim, self.hidden_dim)
                    layers[str(i) + '_a'] = nn.LeakyReLU(negative_slope=0.05)
                layers[str(self.num_decoder_layer)] = nn.Linear(self.hidden_dim, self.x_dim)
                return nn.Sequential(layers)


class GAE(object):
    def __init__(self, n, d, x_dim, seed=0, num_encoder_layer=1, num_decoder_layer=1, hidden_dim=5, latent_dim=1,
                 l1_graph_penalty=0, lr=0.001, device='cuda:0', n_feature=1):
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
        self.net = Net(self.n, self.d, self.x_dim, self.num_encoder_layer, self.num_decoder_layer, self.hidden_dim,
                       self.latent_dim, device=device, seed=seed).to(self.device)

        # for name, param in self.net.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        self.n_features = n_feature

    def _get_mse(self, X, output):
        return torch.square(torch.norm(X - output))

    def _get_loss(self, X, output):
        self.h = torch.trace(torch.matrix_exp(self.net.W_prime * self.net.W_prime)) - self.d
        self.mse_loss = self._get_mse(X, output)
        return 0.5 / self.n * self.mse_loss + self.l1_graph_penalty * torch.norm(self.net.W_prime, p=1) \
               + self.alpha * self.h + \
               0.5 * self.rho * self.h * self.h

    def _process_W_prime(self):
        with torch.no_grad():
            self.net.W_prime = self.net.W_prime.fill_diagonal_(0.)
            if self.n_features > 1:
                self.net.W_prime[:, :self.n_features] = 0.0

    def _train(self, X, rho, alpha, optim):

        self.rho = rho
        self.alpha = alpha
        samples = torch.FloatTensor(X).to(self.device)
        output = self.net.forward(samples)
        loss = self._get_loss(samples, output)
        optim.zero_grad()
        loss.backward()
        optim.step()
        self._get_loss(samples, output)
        self._process_W_prime()
        return loss.item(), self.mse_loss.detach().cpu().numpy(), self.h.detach().cpu().numpy(), self.net.W_prime.detach().cpu().numpy()