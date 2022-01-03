from utils import config_utils, utils, synthetic_dataset, GAE_trainer, analyze_utils, GAES_trainer, AAE_trainer
from models import GAE
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    parser = config_utils.get_args()
    args = parser.parse_args()
    options = vars(args)

    # Reproducibility
    utils.set_seed(options['seed'])

    # Get dataset
    print('Starting generate synthetic data')
    dataset = synthetic_dataset.SyntheticDataset(options['n'] * 2200, options['d'], options['graph_type'], options['degree'], options['sem_type'],
                               options['noise_scale'], options['dataset_type'], options['x_dim'], options['alpha_cos'])
    print('Finish generating synthetic data')
    print('Starting split synthetic data')
    df_train, df_eval, df_test, df_eval_set, df_train_cf, df_eval_cf, df_test_cf, df_eval_set_cf = utils.get_samples(dataset)
    print('Initial GAE')
    gae = GAE.GAE(len(df_train), options['d'], options['x_dim'], options['seed'], options['num_encoder_layers'],
                        options['num_decoder_layers'],
                        options['hidden_size'], options['latent_dim'], options['l1_graph_penalty'], options['learning_rate'], options['device'])

    gae_trainer = GAE_trainer.GAETrainner(options['init_rho'], options['rho_thres'], options['h_thres'], options['rho_multiply'],
                              options['init_iter'], options['learning_rate'], options['h_tol'],
                              options['early_stopping'], options['early_stopping_thres'])
    W_est = gae_trainer.train(gae, df_train.iloc[:, :-3].values.reshape(-1, options['d'], 1), dataset.W,
                              options['graph_thres'],
                              options['max_iter'], options['iter_step'])
    W_est[:, 0] = 0.0

    # Plot raw recovered graph
    analyze_utils.plot_recovered_graph(W_est, dataset.W,
                         save_name='output/raw_recovered_graph.png')

    W_est_old = W_est.copy()
    W_est = W_est / np.max(np.abs(W_est))  # Normalize
    W_est[np.abs(W_est) < options['graph_thres']] = 0  # Thresholding

    analyze_utils.plot_recovered_graph(W_est, dataset.W,
                         save_name='output/thresholded_recovered_graph.png')
    results_thresholded = analyze_utils.count_accuracy(dataset.W, W_est)

    # lst_lr = [1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7, 1e-6]
    lst_lr = [1e-7]
    lst_ratio = [0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1]
    # lst_ratio = [0.995]

    print('Start training GAES')
    gaes = GAES_trainer.GAESTrainner(gae.net.encoder,
                                     df_train.iloc[:, :-3].values.reshape(-1, options['d'], options['x_dim']),
                                     W_est_old,
                                     max_epoch=200, n=len(df_train), d=options['d'], device=options['device'])
    gaes.train()
    scaler = StandardScaler()
    for en_lr in lst_lr:
        for ratio in lst_ratio:
            print('-'*60)
            print(f'results for lr:{en_lr}, ratio:{ratio}')

            print('Start pretrain AAE')
            train_iter, eval_iter, scaler = utils.pretrain_split(df_train, df_eval, scaler)
            aae_trainer = AAE_trainer.AAETrainer(options['d']-1, options['aae_hidden_dim'], options['aae_pretrain_epochs'], quantile=ratio, device=options['device'], alpha=options['aae_alpha'], beta=options['aae_beta'])
            aae_trainer._train(train_iter, eval_iter, pretrain=1)
            print('Pre-training results')
            df_org = utils.get_pretrain_result(gaes, aae_trainer, df_test, df_test_cf, 1, scaler)
            print('Start retrain AAE')
            train_iter, eval_iter = utils.retrain_split(gaes, df_train, df_eval, scaler, device=options['device'])
            aae_trainer.epochs = options['aae_retrain_epochs']
            aae_trainer._train(train_iter, eval_iter, pretrain=0, ae_lr=en_lr, disc_lr=options['discriminator_retrain_lr'])
            df_ad = utils.get_retrain_result(gaes, aae_trainer, df_test, df_test_cf, 1, scaler)
            utils.get_fairness_result(df_org, df_ad, cf=1)
    print('done')

if __name__ == '__main__':
    main()