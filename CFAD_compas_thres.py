from utils import compas_config, utils, GAE_trainer, analyze_utils, GAES_trainer, AAE_trainer
from models import GAE
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch


def main():
    parser = compas_config.get_args()
    args = parser.parse_args()
    options = vars(args)

    # Reproducibility
    utils.set_seed(options['seed'])

    # Get dataset
    print('Starting preprocessing compas dataset')
    df_train, df_test = utils.compas_preprocessing(n_train=options['n_train'], n_test=options['n_test'])
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

    lst_ratio = [0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
    # lst_ratio = [0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
    # lst_ratio =[0.7]
    lst_lr = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    # lst_alpha = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    lst_alpha = [1]
    # lst_ratio.reverse()
    for alpha in lst_alpha:
        for ratio in lst_ratio:
            print('-' * 60)
            print(f'results for lr:{alpha}, ratio:{ratio}')
            print('Start pretrain AAE')
            utils.set_seed(options['seed'])
            scaler = StandardScaler()
            train_iter, eval_iter, scaler = utils.pretrain_split(df_train.iloc[:int(0.9 * len(df_train))],
                                                                 df_train.iloc[int(0.9 * len(df_train)):], scaler, 1)
            aae_trainer = AAE_trainer.AAETrainer(options['d'] - 1, options['aae_hidden_dim'], options['aae_pretrain_epochs'],
                                                 quantile=ratio, device=options['device'],
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