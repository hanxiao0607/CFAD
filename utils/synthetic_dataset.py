import logging
import numpy as np
import networkx as nx


class SyntheticDataset(object):
    """
    Referred from:
    - - https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, graph_type, degree, sem_type, noise_scale=1.0,
                 dataset_type='nonlinear_1', x_dim=1, alpha_cos=3):
        self.n = n
        self.d = d + 1
        self.graph_type = graph_type
        self.degree = degree
        self.sem_type = sem_type
        self.noise_scale = noise_scale
        self.dataset_type = dataset_type
        self.x_dim = x_dim
        self.w_range = (0.5, 1.0)
        self.alpha_cos = alpha_cos

        self._setup()
        self.train_test_split()
        self._logger.debug('Finished setting up dataset class')

    def _setup(self):
        self.W, self.W_all = SyntheticDataset.simulate_random_dag(self.d, self.degree,
                                                                  self.graph_type, self.w_range, self.alpha_cos)

        self.X, self.X_cf, self.y, self.y_cf, self.noise = SyntheticDataset.simulate_sem(self.W_all, self.n,
                                                                                         self.sem_type,
                                                                                         self.noise_scale,
                                                                                         self.dataset_type, self.x_dim,
                                                                                         self.alpha_cos)

    @staticmethod
    def simulate_random_dag(d, degree, graph_type, w_range, alpha_cos):
        """Simulate random DAG with some expected degree.
        Args:
            d: number of nodes
            degree: expected node degree, in + out
            graph_type: {erdos-renyi, barabasi-albert, full}
            w_range: weight range +/- (low, high)
        Returns:
            W: weighted DAG
        """
        if graph_type == 'erdos-renyi':
            prob = float(degree) / (d - 1)
            B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
        elif graph_type == 'barabasi-albert':
            m = int(round(degree / 2))
            B = np.zeros([d, d])
            bag = [0]
            for ii in range(1, d):
                dest = np.random.choice(bag, size=m)
                for jj in dest:
                    B[ii, jj] = 1
                bag.append(ii)
                bag.extend(dest)
        elif graph_type == 'full':  # ignore degree, only for experimental use
            B = np.tril(np.ones([d, d]), k=-1)
        else:
            raise ValueError('unknown graph type')
        # random permutation

        P = np.random.permutation(np.eye(d, d))  # permutes first axis only
        B_perm = P.T.dot(B).dot(P)
        B_perm[:, 0] = 0.0  # set first column to zero
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        U[np.random.rand(d, d) < 0.5] *= -1
        W = (B_perm != 0).astype(float) * U
        W[-1, :] = 0.0
        val_rad = np.append(np.random.uniform(-1, -0.5, 10), np.random.uniform(0.5, 1, 10))
        np.random.shuffle(val_rad)
        for i in range(1, d // 3):
            W[0, i * 3] = val_rad[i]
        W[0, -1] = alpha_cos / 2
        return W[:-1, :-1], W

    @staticmethod
    def simulate_sem(W, n, sem_type, noise_scale=1.0, dataset_type='nonlinear_1', x_dim=1, alpha_cos=3):
        """Simulate samples from SEM with specified type of noise.
        Args:
            W: weigthed DAG
            n: number of samples
            sem_type: {linear-gauss,linear-exp,linear-gumbel}
            noise_scale: scale parameter of noise distribution in linear SEM
        Returns:
            X: [n,d] sample matrix
        """
        G = nx.DiGraph(W)
        d = W.shape[0]
        X = np.zeros([n, d, x_dim])
        X_cf = np.zeros([n, d, x_dim])
        lst_noise = []
        ind_ab = 5
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for ind, j in enumerate(ordered_vertices):
            if ind == 0:
                parents = list(G.predecessors(j))
                assert parents == [], 'Parents should be empty!'
                if sem_type == 'linear-gauss':
                    val_or = np.random.binomial(1, 0.5, n) * 2 - 1
                    lst_noise.append(val_or)
                    val_cf = val_or * (-1)
                    X[:, j, 0] = val_or
                    X_cf[:, j, 0] = val_cf
                else:
                    raise NotImplementedError
            elif j == d - 1:
                parents = list(G.predecessors(j))
                eta = (X[:, parents, 0]).dot(W[parents, j])
                eta_cf = (X_cf[:, parents, 0]).dot(W[parents, j])
                X[:, j, 0] = eta
                X_cf[:, j, 0] = eta_cf
            else:
                parents = list(G.predecessors(j))
                if dataset_type == 'nonlinear_1':
                    eta = alpha_cos * np.cos(X[:, parents, 0] + 1).dot(W[parents, j])
                    eta_cf = alpha_cos * np.cos(X_cf[:, parents, 0] + 1).dot(W[parents, j])
                elif dataset_type == 'nonlinear_2':
                    eta = (X[:, parents, 0] + 0.5).dot(W[parents, j])
                    eta_cf = (X_cf[:, parents, 0] + 0.5).dot(W[parents, j])
                elif dataset_type == 'nonlinear_3':  # Combined version of nonlinear_1 and nonlinear_2
                    eta = np.cos(X[:, parents, 0] + 1).dot(W[parents, j]) + 0.5
                    eta_cf = np.cos(X_cf[:, parents, 0] + 1).dot(W[parents, j]) + 0.5
                else:
                    raise ValueError('Unknown linear data type')

                if sem_type == 'linear-gauss':
                    if dataset_type == 'nonlinear_1':
                        noise = np.random.normal(scale=noise_scale, size=n)
                        lst_noise.append(noise)
                        #                         if ind == ind_ab:
                        #                             noise = np.random.normal(scale=noise_scale*3, size=n)
                        X[:, j, 0] = eta + noise
                        X_cf[:, j, 0] = eta_cf + noise
                    elif dataset_type in ('nonlinear_2', 'nonlinear_3'):
                        noise = np.random.normal(scale=noise_scale, size=n)
                        if ind == ind_ab:
                            noise = np.random.normal(scale=noise_scale, size=n)
                        X[:, j, 0] = 2. * np.sin(eta) + eta + noise
                        X_cf[:, j, 0] = 2. * np.sin(eta_cf) + eta_cf + noise
                else:
                    raise NotImplementedError

        if x_dim > 1:
            raise NotImplementedError
        y = X[:, -1, :].reshape(-1)
        y_cf = X_cf[:, -1, :].reshape(-1)
        return X[:, :-1, :], X_cf[:, :-1, :], y, y_cf, lst_noise

    def train_test_split(self, value=0, ratio=0.6):
        self.label = self.y.copy()
        ind = int(len(self.X) * ratio)
        self.label[self.label <= value] = 0
        self.label[self.label > value] = 1
        self.train_X = self.X[:ind][self.label[:ind] == 0]
        self.test_X = self.X[ind:]
        self.test_y = self.y[ind:]
        self.test_label = self.label[ind:]
        self.n_train = len(self.train_X)
