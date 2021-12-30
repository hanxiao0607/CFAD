from models import GAES
import torch
from torch import nn
from tqdm import tqdm


class GAESTrainner(object):
    def __init__(self, encoder, X, W, max_epoch, n, d, device='cuda:0', seed=0, weight_decay=1e-1,
                 name='gaes_checkpoint.pt'):
        self.X = X
        self.W = torch.FloatTensor(W).to(device)
        self.max_epoch = max_epoch

        self.n = n
        self.d = d
        self.device = device
        self.seed = seed

        self.weight_decay = weight_decay

        self.name = name
        self.net = GAES.GAES(encoder, self.W, n, d, len(X[0][0])).to(self.device)
        for param in self.net.encoder.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                          weight_decay=self.weight_decay)
        self.loss = nn.MSELoss()

    def train(self):
        self.net.train()
        best_loss = float('inf')
        for i in range(self.max_epoch):
            epoch_loss = 0
            samples = torch.FloatTensor(self.X).to(self.device)
            self.optimizer.zero_grad()
            samples_hat = self.net(samples)
            loss = self.loss(samples_hat, samples)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            samples.cpu()
            if best_loss > epoch_loss:
                torch.save(self.net, f'saved_models/{self.name}')
                best_loss = epoch_loss
            if i == 0:
                print(f'Initial loss: {epoch_loss}')
        print(f'Final loss: {best_loss}')