import torch
from torch import nn
import numpy as np
import networkx as nx
from collections import OrderedDict


class GAES(nn.Module):
    def __init__(self, encoder, W, n=3000, d=20, x_dim=1, hidden_dim=16, latent_dim=1, device='cuda:0', seed=0, \
                 num_encoder_layer=2, num_decoder_layer=5, graph_thres=0.20):
        super(GAES, self).__init__()
        self.seed = seed

        self.n = n
        self.d = d

        self.device = device
        self.graph_thres = graph_thres
        self.A = W.cpu().numpy()
        self._normalize_A()
        self.W = W.to(device)
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_encoder_layer = num_encoder_layer
        self.num_decoder_layer = num_decoder_layer

        self.encoder = encoder

        self.decoder = self._define_net(encoder=0)

    #         self.ah_weight = nn.Parameter(torch.rand(self.d))
    #         self.ah_weight.requires_grad = True
    #         self.ah_bias = nn.Parameter(torch.rand(self.d))
    #         self.ah_bias.requires_grad = True

    def _normalize_A(self):
        while 1:
            W_est = self.A.copy()
            W_est = W_est / np.max(np.abs(W_est))  # Normalize
            W_est[np.abs(W_est) < self.graph_thres] = 0  # Thresholding
            try:
                self.G = nx.DiGraph(W_est)
                ordered_vertices = list(nx.topological_sort(self.G))
                self.A_norm = torch.FloatTensor(W_est).to(self.device)
                break
            except:
                self.graph_thres += 0.01

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

    def forward(self, X):
        ordered_vertices = list(nx.topological_sort(self.G))
        X_hat = torch.zeros(X.size())
        assert len(ordered_vertices) == self.d
        for ind, j in enumerate(ordered_vertices):
            if ind == 0:
                parents = list(self.G.predecessors(j))
                assert parents == [], 'Parents should be empty!'
                assert j == 0, 'First should be cf!'
                X_hat[:, j, 0] = torch.reshape(X, (-1, self.d, 1)).cpu()[:, j, 0].clone()

            else:
                parents = list(self.G.predecessors(j))
                if len(parents) == 0:
                    X_hat[:, j, 0] = torch.reshape(X, (-1, self.d, 1)).cpu()[:, j, 0].clone()
                elif len(parents) == 1:
                    H = torch.reshape(self.encoder.forward(X), (-1, self.d, 1))
                    #                     H_c = torch.reshape(H[:, parents, 0]*self.A_norm[parents, j], (-1, 1))
                    #                     X_hat[:, j, 0] = torch.flatten(self.decoder.forward(H_c).cpu()).clone()
                    H_c = torch.reshape(H[:, parents, 0], (-1, 1))
                    X_hat[:, j, 0] = (torch.flatten(self.decoder.forward(H_c)) * self.A_norm[parents, j]).cpu().clone()
                else:
                    H = torch.reshape(self.encoder.forward(X), (-1, self.d, 1))
                    #                     H_c = torch.reshape(H[:, parents, 0].matmul(self.A_norm[parents, j]), (-1, 1))
                    #                     X_hat[:, j, 0] = torch.flatten(self.decoder.forward(H_c).cpu()).clone()
                    H_c = torch.reshape(H[:, parents, 0], (-1, 1))
                    X_hat[:, j, 0] = torch.reshape(self.decoder.forward(H_c), (-1, len(parents))).matmul(
                        self.A_norm[parents, j]).cpu().clone()
        #                     X_hat[:, j, 0] = torch.reshape(X, (-1, self.d, 1)).cpu()[:, j, 0].clone()
        return X_hat.to(self.device)

    def get_result(self, X, do=0, p=0):
        ordered_vertices = list(nx.topological_sort(self.G))
        X_hat = self.forward(X)
        X_noise = (X - X_hat).cpu()
        if do:
            X_do = X.clone()
            X_do[:, 0, 0] = X_do[:, 0, 0] * (-1)
            X_do_hat = torch.zeros(X_do.size())
            assert len(ordered_vertices) == self.d
            for ind, j in enumerate(ordered_vertices):
                if ind == 0:
                    parents = list(self.G.predecessors(j))
                    assert parents == [], 'Parents should be empty!'
                    assert j == 0, 'First should be cf!'
                    X_do_hat[:, j, 0] = torch.reshape(X_do, (-1, self.d, 1)).cpu()[:, j, 0].clone()
                else:
                    parents = list(self.G.predecessors(j))
                    if len(parents) == 0:
                        X_do_hat[:, j, 0] = torch.reshape(X_do, (-1, self.d, 1)).cpu()[:, j, 0].clone()
                    elif len(parents) == 1:
                        H = torch.reshape(self.encoder.forward(torch.reshape(X_do_hat.cuda(), (-1, 1))),
                                          (-1, self.d, 1))
                        #                         H_c = torch.reshape(H[:, parents, 0]*self.A_norm[parents, j], (-1, 1))
                        #                         X_do_hat[:, j, 0] = torch.flatten(self.decoder.forward(H_c).cpu()).clone() + X_noise[:, j, 0]
                        H_c = torch.reshape(H[:, parents, 0], (-1, 1))
                        X_do_hat[:, j, 0] = torch.flatten(
                            (self.decoder.forward(H_c) * self.A_norm[parents, j]).cpu()).clone() + X_noise[:, j, 0]
                    else:
                        H = torch.reshape(self.encoder.forward(torch.reshape(X_do_hat.cuda(), (-1, 1))),
                                          (-1, self.d, 1))
                        #                         H_c = torch.reshape(H[:, parents, 0].matmul(self.A_norm[parents, j]), (-1, 1))
                        #                         X_do_hat[:, j, 0] = torch.flatten(self.decoder.forward(H_c).cpu()).clone() + X_noise[:, j, 0]
                        H_c = torch.reshape(H[:, parents, 0], (-1, 1))
                        X_do_hat[:, j, 0] = torch.reshape(self.decoder.forward(H_c), (-1, len(parents))).matmul(
                            self.A_norm[parents, j]).cpu().clone() + X_noise[:, j, 0]

        return X_do_hat.to(self.device)