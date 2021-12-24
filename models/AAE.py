import torch.nn as nn
from torch.autograd import Function


class GRL(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class AAE(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, hid_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hid_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(hid_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2)
        )

    def forward(self, x, alpha):
        z = self.encoder(x)
        y = GRL.apply(z, alpha)
        xhat = self.decoder(z)
        return z, xhat, y