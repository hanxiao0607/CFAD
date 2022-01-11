import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from models import AAE


class AAETrainer(nn.Module):
    def __init__(self, input_dim, hid_dim, num_epochs, quantile=0.95, device='cuda:0', alpha=1, beta=1):
        super().__init__()
        self.dae = AAE.AAE(input_dim, hid_dim).to(device)
        self.device = device
        self.epochs = num_epochs
        self.alpha = alpha
        self.beta = beta
        self.max_dist = np.inf
        self.quantile = quantile

    def _train(self, train_iter, eval_iter, pretrain=1, ae_lr=1e-6, disc_lr=1e-5):
        if pretrain:
            optimizer = torch.optim.Adam([{'params': self.dae.encoder.parameters()},
                                          {'params': self.dae.decoder.parameters()},
                                          ], lr=1e-5, weight_decay=1e-6)
        else:
            optimizer = torch.optim.Adam([{'params': self.dae.encoder.parameters(), 'lr': ae_lr},
                                          {'params': self.dae.decoder.parameters(), 'lr': ae_lr},
                                          {'params': self.dae.discriminator.parameters()},
                                          ], lr=disc_lr, weight_decay=1e-6)

        loss_mse = nn.MSELoss(reduction='mean')
        loss_ce = nn.CrossEntropyLoss()
        min_loss = np.inf
        for epoch in range(self.epochs):
            loss_mse_lst = []
            loss_ce_lst = []
            self.train()
            for batch in train_iter:
                optimizer.zero_grad()
                x = batch[0].to(self.device)
                do = batch[1].to(self.device)
                _, decoded, do_hat = self.dae.forward(x, self.alpha)
                loss1 = loss_mse(decoded, x)
                loss2 = loss_ce(do_hat, do)
                if pretrain:
                    loss = loss1
                else:
                    loss = loss1 + loss2 * self.beta
                loss.backward()
                optimizer.step()
                loss_mse_lst.append(loss1.item())
                loss_ce_lst.append(loss2.item())
            mse_loss = np.mean(loss_mse_lst)
            ce_loss = np.mean(loss_ce_lst)
            # print(f'Training loss for epoch:{epoch}, loss_mse: {mse_loss}, loss_ce: {ce_loss}')

            loss_mse_lst = []
            loss_ce_lst = []
            lst_dist = []
            self.eval()
            for batch in eval_iter:
                x = batch[0].to(self.device)
                do = batch[1].to(self.device).long()
                _, decoded, do_hat = self.dae.forward(x, self.device)
                dist = ((decoded - x) ** 2).sum(axis=1)
                # if pretrain:
                #     dist = ((decoded - x) ** 2).sum(axis=1)
                # else:
                #     idx = (do == 0).nonzero().view(-1)
                #     dist = ((decoded[idx] - x[idx]) ** 2).sum(axis=1)
                lst_dist.extend(dist.detach().cpu().numpy().reshape(-1))
                loss1 = loss_mse(decoded, x)
                loss2 = loss_ce(do_hat, do)
                loss = loss1 + loss2
                loss_mse_lst.append(loss1.item())
                loss_ce_lst.append(loss2.item())
            mse_loss = np.mean(loss_mse_lst)
            ce_loss = np.mean(loss_ce_lst)
            # print(f'Evaluation loss for epoch:{epoch}, loss_mse: {mse_loss}, loss_ce: {ce_loss}')
            if pretrain:
                if mse_loss < min_loss:
                    torch.save(self.state_dict(), 'saved_models/aae_checkpoint.pt')
                    self.max_dist = np.quantile(np.array(lst_dist), self.quantile)
            else:
                #                 if mse_loss*self.beta+ce_loss < min_loss:
                #                 if mse_loss < min_loss:
                torch.save(self.state_dict(), 'saved_models/aae_checkpoint.pt')
                self.max_dist = np.quantile(np.array(lst_dist), self.quantile)

    def _evaluation(self, test_iter, y, r=1):
        self.load_state_dict(torch.load('saved_models/aae_checkpoint.pt'))
        self.eval()
        lst_loss = []
        lst_pred = []
        lst_dist = []
        criterion = nn.MSELoss(reduction='none')
        for batch in test_iter:
            x = batch.to(self.device)
            _, decoded, _ = self.dae.forward(x, self.alpha)
            dist = ((decoded - x) ** 2).sum(axis=1)
            loss = criterion(decoded, x)
            loss = torch.mean(loss, dim=1).detach().cpu().numpy()
            lst_loss.extend(loss)
            lst_dist.extend(dist.detach().cpu().numpy())
            lst_pred.extend([1 if i > r else 0 for i in dist])
        loss_mean = np.mean(lst_loss)
        print(f'Final Evalution loss for {loss_mean}')
        print('-' * 60)

        return lst_pred, np.array(lst_dist).reshape(-1, 1)