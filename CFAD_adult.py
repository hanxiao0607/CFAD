import pandas as pd

from utils import adult_config, utils, GAE_trainer, analyze_utils, GAES_trainer, AAE_trainer
from models import GAE
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


def main():
    parser = adult_config.get_args()
    args = parser.parse_args()
    options = vars(args)

    # Reproducibility
    utils.set_seed(options['seed'])

    # Get dataset
    print('Starting preprocessing adult dataset')
    df_train, df_test = utils.adult_preprocessing(n_train=options['n_train'], n_test=options['n_test'])
    df_train.to_csv('data/adult_train.csv')
    df_test.to_csv('data/adult_test.csv')
    print('Finish preprocessing')
    gae = GAE.GAE(len(df_train), options['d'], options['x_dim'], options['seed'], options['num_encoder_layers'],
                  options['num_decoder_layers'],
                  options['hidden_size'], options['latent_dim'], options['l1_graph_penalty'], options['learning_rate'],
                  options['device'], options['n_feature'])

    gae_trainer = GAE_trainer.GAETrainner(options['init_rho'], options['rho_thres'], options['h_thres'],
                                          options['rho_multiply'],
                                          options['init_iter'], options['learning_rate'], options['h_tol'],
                                          options['early_stopping'], options['early_stopping_thres'])
    W_est = gae_trainer.train(gae, df_train.iloc[:, :-1].values.reshape(-1, options['d'], 1).astype(np.float32), None,
                              options['graph_thres'],
                              options['max_iter'], options['iter_step'])
    W_est[:, 0] = 0.0

    W_est_old = W_est.copy()
    W_est = W_est / np.max(np.abs(W_est))  # Normalize
    W_est[np.abs(W_est) < options['graph_thres']] = 0  # Thresholding

    analyze_utils.plot_recovered_graph(W_est, W_est,
                                       save_name='output/thresholded_recovered_graph.png')

    analyze_utils.plot_single_graph(W_est, save_name='output/adult_A.png')

    print('Start training GAES')
    gaes = GAES_trainer.GAESTrainner(gae.net.encoder,
                                     df_train.iloc[:, :-1].values.reshape(-1, options['d'], options['x_dim']).astype(
                                         np.float32), W_est_old,
                                     max_epoch=200, n=len(df_train), d=options['d'], device=options['device'],
                                     weight_decay=0, adult=1)
    gaes.train()
    test_do = gaes.net.get_result(
        torch.Tensor(df_test.iloc[:, :-1].values.astype(np.float32).reshape(len(df_test), -1, 1)).to(options['device']),
        do=1).detach().cpu().numpy().reshape(len(df_test), -1)[:, 1:]
    pd.DataFrame(test_do).to_csv('data/adult_do.csv')
    print('Start pretrain AAE')
    scaler = StandardScaler()
    train_iter, eval_iter, scaler = utils.pretrain_split(df_train.iloc[:int(0.9 * len(df_train))],
                                                         df_train.iloc[int(0.9 * len(df_train)):], scaler, 1)
    aae_trainer = AAE_trainer.AAETrainer(options['d'] - 1, options['aae_hidden_dim'], options['aae_pretrain_epochs'],
                                         quantile=options['quantile'], device=options['device'],
                                         alpha=options['aae_alpha'], beta=options['aae_beta'])
    aae_trainer._train(train_iter, eval_iter, pretrain=1)
    print('Pre-training results')
    df_org = utils.get_pretrain_results_adult(aae_trainer, df_test, test_do)
    print('Start retrain AAE')
    train_iter, eval_iter = utils.retrain_split(gaes, df_train.iloc[:int(0.9 * len(df_train))],
                                                df_train.iloc[int(0.9 * len(df_train)):], scaler,
                                                device=options['device'], adult=1)
    aae_trainer.epochs = options['aae_retrain_epochs']
    aae_trainer._train(train_iter, eval_iter, pretrain=0, ae_lr=options['ae_retrain_lr'],
                       disc_lr=options['discriminator_retrain_lr'])
    df_ad = utils.get_retrain_results_adult(aae_trainer, df_test, test_do)
    utils.get_fairness_result(df_org, df_ad, cf=0)
    print('done')



if __name__ == '__main__':
    main()