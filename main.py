import torch
from models import gae

def main():
    n, d = 3000, 20

    model = gae.GAE(n, d, x_dim=1)

    print(model.w_prime)
    # print('model.sess: {}'.format(model.sess))
    # print('model.train_op: {}'.format(model.train_op))
    # print('model.loss: {}'.format(model.loss))
    # print('model.mse_loss: {}'.format(model.mse_loss))
    # print('model.h: {}'.format(model.h))
    # print('model.W_prime: {}'.format(model.W_prime))
    # print('model.X: {}'.format(model.X))
    # print('model.rho: {}'.format(model.rho))
    # print('model.alpha: {}'.format(model.alpha))
    # print('model.lr: {}'.format(model.lr))

    print('done')

if __name__ == '__main__':
    main()