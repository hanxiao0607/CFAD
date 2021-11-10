import torch
import numpy as np
from models import gae
from utils import config_utils, utils, synthetic_dataset, trainer


def main():
    args = config_utils.get_args()

    # Reproducibility
    utils.set_seed(args.seed)

    # Get dataset
    dataset = synthetic_dataset.SyntheticDataset(args.n, args.d, args.graph_type, args.degree, args.sem_type,
                               args.noise_scale, args.dataset_type, args.x_dim)
    model = gae.GAE(args.n, args.d, args.x_dim, args.seed, args.num_encoder_layers, args.num_decoder_layers,
                args.hidden_size, args.latent_dim, args.l1_graph_penalty, args.use_float64)

    gae_trainer = trainer.GAETrainner(args.init_rho, args.rho_thres, args.h_thres, args.rho_multiply,
                        args.init_iter, args.learning_rate, args.h_tol,
                        args.early_stopping, args.early_stopping_thres)
    W_est = gae_trainer.train(model, dataset.X, dataset.W, args.graph_thres,
                          args.max_iter, args.iter_step)



    print('done')

if __name__ == '__main__':
    main()