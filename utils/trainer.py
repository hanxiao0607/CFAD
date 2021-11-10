import numpy as np
import torch
from . import utils


class GAETrainner(object):

    def __init__(self, init_rho, rho_thres, h_thres, rho_multiply, init_iter,
                 learning_rate, h_tol, early_stopping, early_stopping_thres):
        self.init_rho = init_rho
        self.rho_thres = rho_thres
        self.h_thres = h_thres
        self.rho_multiply = rho_multiply
        self.init_iter = init_iter
        self.learning_rate = learning_rate
        self.h_tol = h_tol
        self.early_stopping = early_stopping
        self.early_stopping_thres = early_stopping_thres

    def train(self, model, X, W, graph_thres, max_iter, iter_step, name='init_model.pt'):
        rho, alpha, h, h_new = self.init_rho, 0.0, np.inf, np.inf
        prev_W_est, prev_mse = None, float('inf')

        for i in range(1, max_iter + 1):
            while rho < self.rho_thres:
                loss_new, mse_new, h_new, W_new = self.train_step(model, iter_step, X, rho, alpha)
                if h_new > self.h_thres * h:
                    rho *= self.rho_multiply
                else:
                    break

            if self.early_stopping:
                if mse_new / prev_mse > self.early_stopping_thres and h_new <= 1e-7:
                    return prev_W_est
                else:
                    prev_W_est = W_new
                    prev_mse = mse_new

            W_est, h = W_new, h_new
            alpha += rho * h
            if h <= self.h_tol and i > self.init_iter:
                print(f'Early stopping at {i}-th iteration')
                break

        torch.save(model, f'./saved_models/{name}')

        return W_est

    def train_step(self, model, iter_step, X, rho, alpha):
        for _ in range(iter_step):
            curr_loss, curr_mse, curr_h, curr_W = model._train(X, rho, alpha, self.learning_rate)
        return curr_loss, curr_mse, curr_h, curr_W
