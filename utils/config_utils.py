from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()

    ##### General settings #####
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed')

    ##### Dataset settings #####
    parser.add_argument('--n',
                        type=int,
                        default=10000,
                        help='Number of samples')

    parser.add_argument('--d',
                        type=int,
                        default=20,
                        help='Number of nodes')

    parser.add_argument('--graph_type',
                        type=str,
                        default='erdos-renyi',
                        help='Type of graph')

    parser.add_argument('--degree',
                        type=int,
                        default=5,
                        help='Degree of graph')

    parser.add_argument('--sem_type',
                        type=str,
                        default='linear-gauss',
                        help='Type of sem')

    parser.add_argument('--noise_scale',
                        type=float,
                        default=1.0,
                        help='Variance of Gaussian Noise')

    parser.add_argument('--dataset_type',
                        type=str,
                        default='nonlinear_1',
                        help='Choose between nonlinear_1, nonlinear_2, nonlinear_3')

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
                        default=16,
                        help='Hidden size for NN layers')

    parser.add_argument('--latent_dim',
                        type=int,
                        default=1,
                        help='Latent dimension for autoencoder')

    parser.add_argument('--l1_graph_penalty',
                        type=float,
                        default=1.0,
                        help='L1 penalty for sparse graph. Set to 0 to disable')

    parser.add_argument('--use_float64',
                        type=bool,
                        default=True,
                        help='Whether to use tf.float64 or tf.float32 during training')

    ##### GAE Training settings #####
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help='Learning rate for Adam optimizer')

    parser.add_argument('--max_iter',
                        type=int,
                        default=20,
                        help='Number of iterations for training/optimization')

    parser.add_argument('--iter_step',
                        type=int,
                        default=300,
                        help='Number of steps for each iteration')

    parser.add_argument('--init_iter',
                        type=int,
                        default=5,
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
                        default=0.2,
                        help='Threshold to filter out small values in graph')

    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='Device type')

    parser.add_argument('--alpha_cos',
                        type=int,
                        default=3,
                        help='Times for cosine value')

    parser.add_argument('--aae_hidden_dim',
                        type=int,
                        default=8,
                        help='Hidden dimension for AAE')

    parser.add_argument('--aae_pretrain_epochs',
                        type=int,
                        default=50,
                        help='AAE max pre-training epochs')

    parser.add_argument('--aae_retrain_epochs',
                        type=int,
                        default=20,
                        help='AAE max re-training epochs')

    parser.add_argument('--aae_alpha',
                        type=float,
                        default=1,
                        help='AAE adversarial parameter')

    parser.add_argument('--aae_beta',
                        type=float,
                        default=1,
                        help='AAE reconstruction parameter')

    parser.add_argument('--ae_retrain_lr',
                         type=float,
                         default=4e-6,
                         help='autoencoder lr for retrain')

    parser.add_argument('--discriminator_retrain_lr',
                        type=float,
                        default=1e-4,
                        help='discriminator lr for retrain')


    return parser