from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()

    ##### General settings #####
    parser.add_argument('--env_seed',
                        type=int,
                        default=0,
                        help='Env seed')

    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed')

    ##### Dataset settings #####
    parser.add_argument('--n_train',
                        type=int,
                        default=12000,
                        help='Number of training samples')

    parser.add_argument('--n_test',
                        type=int,
                        default=4000,
                        help='Number of testing samples')

    parser.add_argument('--d',
                        type=int,
                        default=14,
                        help='Number of nodes')

    parser.add_argument('--x_dim',
                        type=int,
                        default=1,
                        help='Dimension of vector for X')

    ##### Model settings #####
    parser.add_argument('--num_encoder_layers',
                        type=int,
                        default=5,
                        help='Number of hidden layers for encoder')

    parser.add_argument('--num_decoder_layers',
                        type=int,
                        default=5,
                        help='Number of hidden layers for decoder')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=64,
                        help='Hidden size for NN layers')

    parser.add_argument('--latent_dim',
                        type=int,
                        default=1,
                        help='Latent dimension for autoencoder')

    parser.add_argument('--l1_graph_penalty',
                        type=float,
                        default=0.0,
                        help='L1 penalty for sparse graph. Set to 0 to disable')

    parser.add_argument('--use_float64',
                        type=bool,
                        default=True,
                        help='Whether to use tf.float64 or tf.float32 during training')

    ##### GAE Training settings #####
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-5,
                        help='Learning rate for Adam optimizer')

    parser.add_argument('--max_iter',
                        type=int,
                        default=200,
                        help='Number of iterations for training/optimization')

    parser.add_argument('--iter_step',
                        type=int,
                        default=300,
                        help='Number of steps for each iteration')

    parser.add_argument('--init_iter',
                        type=int,
                        default=2,
                        help='Initial iterations to disable early stopping')

    parser.add_argument('--h_tol',
                        type=float,
                        default=1e-12,
                        help='Tolerance for acyclicity constraint')

    parser.add_argument('--init_rho',
                        type=float,
                        default=1.0,
                        help='Initial value for rho')

    parser.add_argument('--rho_thres',
                        type=float,
                        default=1e+18,
                        help='Threshold for rho')

    parser.add_argument('--h_thres',
                        type=float,
                        default=0.25,
                        help='Threshold for h')

    parser.add_argument('--rho_multiply',
                        type=float,
                        default=10,
                        help='Multiplication to amplify rho each time')

    parser.add_argument('--early_stopping',
                        type=bool,
                        default=True,
                        help='Whether to use early stopping')

    parser.add_argument('--early_stopping_thres',
                        type=float,
                        default=1.15,
                        help='Threshold ratio for early stopping')

    parser.add_argument('--graph_thres',
                        type=float,
                        default=0.3,
                        help='Threshold to filter out small values in graph')

    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='Device type')

    parser.add_argument('--n_feature',
                        type=int,
                        default=4,
                        help='Features without father')


    parser.add_argument('--aae_hidden_dim',
                        type=int,
                        default=8,
                        help='Hidden dimension for AAE')

    parser.add_argument('--aae_pretrain_epochs',
                        type=int,
                        default=100,
                        help='AAE max pre-training epochs')

    parser.add_argument('--aae_retrain_epochs',
                        type=int,
                        default=20,
                        help='AAE max re-training epochs')

    parser.add_argument('--aae_alpha',
                        type=float,
                        default=0.1,
                        help='AAE adversarial parameter')

    parser.add_argument('--aae_beta',
                        type=float,
                        default=1,
                        help='AAE reconstruction parameter')

    parser.add_argument('--ae_retrain_lr',
                         type=float,
                         default=1e-7,
                         help='autoencoder lr for retrain')

    parser.add_argument('--discriminator_retrain_lr',
                        type=float,
                        default=1e-4,
                        help='discriminator lr for retrain')

    parser.add_argument('--quantile',
                        type=float,
                        default=0.95,
                        help='quantile for reconstruction error')


    return parser