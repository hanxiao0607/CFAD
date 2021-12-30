"""
Code for Deep Clustering Fair Outlier Detection
Source: https://github.com/brandeis-machine-learning/FairOutlierDetection/blob/5a6ccc997274619d2cab95507969b535312c4375/code/train.py
Date: 11/2020
"""
import pandas as pd
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from model import DCFOD
import warnings
from utils import utils, synthetic_dataset, config_utils
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=FutureWarning)

# -----indicate which gpu to use for training, devices list will be used in training with DataParellel----- #
gpu = '1'
if gpu == '0':
    cuda = torch.device('cuda:0')
    devices = [0, 1, 2, 3]
elif gpu == '1':
    cuda = torch.device('cuda:1')
    devices = [1, 2, 3, 0]
elif gpu == '2':
    cuda = torch.device('cuda:2')
    devices = [2, 3, 0, 1]
elif gpu == '3':
    cuda = torch.device('cuda:3')
    devices = [3, 0, 1, 2]
else:
    raise NameError('no more GPUs')


def set_seed(seed):
    print(f"setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False


def acc(Y, dist):
    """
    Calculate the AUC, Fgap, and Frank
    Args:
        dset: dataset
        Y: ground truth outlier label
        dist: distance to cluster centers
    Returns: AUC, Fgap, Frank
    """

    outlier_score, position = torch.min(dist, dim=1)
    for i in range(dist.shape[1]):
        pos = list(x for x in range(len(outlier_score)) if position[x] == i)
        if len(outlier_score[pos]) != 0:
            max_dist = max(outlier_score[pos])
            outlier_score[pos] = torch.div(outlier_score[pos], max_dist).to(cuda)
    if len(set(Y)) > 1:
        AUC = roc_auc_score(Y, outlier_score.data.cpu().numpy())
        PR = average_precision_score(Y, outlier_score.data.cpu().numpy())
    else:
        AUC = -1
        PR = -1
    return AUC, PR, outlier_score.data.cpu().numpy()


def target_distribution(q):
    """
    Calculate the auxiliary distribution with the original distribution
    Args:
        q: original distribution
    Returns: auxiliary distribution
    """
    weight = (q ** 2) / q.sum(0)
    return torch.div(weight.t(), weight.sum(1)).t().data


def kld(q, p):
    """
    KL-divergence
    Args:
        q: original distribution
        p: auxiliary distribution
    Returns: the similarity between two probability distributions
    """
    return torch.sum(p * torch.log(p / q).to(cuda), dim=-1)


def getTDistribution(model, x):
    """
    Obtain the distance to centroid for each instance, and calculate the weight module based on that
    Args:
        model: DCFOD
        x: embedded x
    Returns: weight module, clustering distribution
    """

    # dist, dist_to_centers = model.module.getDistanceToClusters(x)
    dist, dist_to_centers = model.getDistanceToClusters(x)

    # -----find the centroid for each instance, with their distance in between----- #
    outlier_score, centroid = torch.min(dist_to_centers, dim=1)

    # -----for each instance, calculate a score
    # by the outlier_score divided by the furtherest instance in the centroid----- #
    for i in range(dist_to_centers.shape[1]):
        pos = list(x for x in range(len(outlier_score)) if centroid[x] == i)
        if len(outlier_score[pos]) != 0:
            max_dist = max(outlier_score[pos])
            outlier_score[pos] = torch.div(outlier_score[pos], max_dist).to(cuda)
    sm = nn.Softmax(dim=0).to(cuda)
    weight = sm(outlier_score.neg())

    # -----calculate the clustering distribution with the distance----- #
    q = 1.0 / (1.0 + (dist / model.alpha))
    q = q ** (model.alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, 1)).t()
    return weight, q


def clustering(model, mbk, x):
    """
    Initialize cluster centroids with minibatch Kmeans
    Args:
        model: DCFOD
        mbk: minibatch Kmeans
        x: embedded x
    Returns: N/A
    """
    model.eval()
    x_e = model(x.float())
    mbk.partial_fit(x_e.data.cpu().numpy())
    model.cluster_centers = mbk.cluster_centers_  # keep the cluster centers
    model.clusterCenter.data = torch.from_numpy(model.cluster_centers).to(cuda)


def Train(model, train_input, labels, attribute, epochs, batch, with_weight=False, ks=8, kf=100):
    """
    Train DCFOD in minibatch
    Args:
        model: DCFOD
        train_input: input data
        labels: ground truth outlier score, which will not be used during training
        attribute: sensitive attribute subgroups
        epochs: total number of iterations
        batch: minibatch size
        with_weight: if training with weight
        ks: hyperparameter for self-reconstruction loss
        kf: hyperparameter for fairness-adversarial loss
    Returns: AUC, Fgap, Frank
    """
    model.train()
    mbk = MiniBatchKMeans(n_clusters=model.num_classes, n_init=20, batch_size=batch)
    got_cluster_center = False
    running_loss = 0.0
    fair_loss = 0.0
    lr_cluster = 0.0001
    lr_discriminator = 0.00001
    lr_sae = 0.00001

    optimizer = optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
        {'params': model.discriminator.parameters(), 'lr': lr_discriminator},
        {'params': model.clusterCenter, 'lr': lr_cluster}
    ], lr=lr_sae, weight_decay=1e-6)
    # optimizer = optim.SGD([
    #     {'params': model.encoder.parameters()},
    #     {'params': model.decoder.parameters()},
    #     {'params': model.discriminator.parameters(), 'lr': lr_discriminator},
    #     {'params': model.clusterCenter, 'lr': lr_cluster}
    # ], lr=lr_sae, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print(f'Learning rate: {lr_cluster}, {lr_sae}, {lr_discriminator}')
    print(f'batch size: {batch}, self_recon: {ks}, fairness: {kf}')

    for epoch in range(epochs):
        for i in range(train_input.shape[0] // batch):
            input_batch = train_input[i * batch: (i + 1) * batch]
            x = torch.tensor(input_batch).float()
            x = x.to(cuda)

            attribute_batch = attribute[i * batch: (i + 1) * batch]
            attribute_batch = torch.tensor(attribute_batch).to(cuda).long()
            # -----use minibatch Kmeans to initialize the cluster centroids for the clustering layer----- #
            if not got_cluster_center:
                # model.module.set_clustering_mode(True)
                model.setClusteringMode(True)
                total = torch.tensor(train_input).to(cuda)
                clustering(model, mbk, total)
                got_cluster_center = True
                # model.module.set_clustering_mode(False)
                model.setClusteringMode(False)
            else:
                model.train()
                x_e, x_de, x_sa = model(x)
                # -----obtain the clustering probability distribution and dynamic weight----- #
                weight, q = getTDistribution(model, x_e)
                if x.shape != x_de.shape:
                    x = np.reshape(x.data.cpu().numpy(), x_de.shape)
                    x = torch.tensor(x).to(cuda)
                p = target_distribution(q)
                clustering_regularizer_loss = kld(q, p)

                self_reconstruction_loss = nn.functional.mse_loss(x_de, x, reduction='none').to(cuda)
                self_reconstruction_loss = torch.sum(self_reconstruction_loss, dim=2)
                self_reconstruction_loss = torch.reshape(self_reconstruction_loss, (self_reconstruction_loss.shape[0],))

                CELoss = nn.CrossEntropyLoss().to(cuda)
                discriminator_loss = CELoss(x_sa, attribute_batch)

                if with_weight:
                    objective = ks * self_reconstruction_loss + kf * discriminator_loss + clustering_regularizer_loss
                    L = objective.mean()
                    # L = torch.sum(torch.mul(objective, weight))
                else:
                    objective = self_reconstruction_loss + clustering_regularizer_loss
                    L = objective.mean()
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
                running_loss += L.data.cpu().numpy()
                fair_loss += discriminator_loss.data.cpu().numpy()

                # -----show loss every 20 mini-batches----- #
                if i % 30 == 29:
                    print(f'[{epoch + 1},     {i + 1}] L:{running_loss / 30:.2f}, FairLoss: {fair_loss / 30:.4f}')
                    running_loss = 0.0
                    fair_loss = 0.0

        if epoch == epochs-1:
            normal_dist = get_abdist(model, train_input, labels)

        scheduler.step()

    print('Done Training.')
    return normal_dist

def get_abdist(model, train_input, Y):
    torch.cuda.empty_cache()
    model.eval()
    model.setValidateMode(True)
    model_input = torch.tensor(train_input).to(cuda)
    xe = model(model_input.float())
    _, dist = model.getDistanceToClusters(xe)
    model.setValidateMode(False)
    return np.quantile(acc(Y, dist)[-1], 0.95)

def validate(model, eval_input, Y, normal_dist):
    """
    check the model performance after one iteration of minibatch training
    Args:
        model: DCFOD
        eval_input: input data
        Y: ground truth outlier labels
    Returns: AUC, Fgap, Frank
    """

    # -----empty cache to save memory for kdd dataset, or have to use DataParellel----- #
    torch.cuda.empty_cache()
    model.eval()

    # -----set model to validate mode, so it only returns the embedded space----- #
    # model.module.setTrainValidateMode(True)
    model.setValidateMode(True)
    model_input = torch.tensor(eval_input).to(cuda)
    xe = model(model_input.float())

    # -----obtain all instances' distance to cluster centroids----- #
    # _, dist = model.module.getDistanceToClusters(x)
    _, dist = model.getDistanceToClusters(xe)

    # -----set to retrieve AUC, Fgap, Frank values in acc function----- #
    AUC, PR, max_dist = acc(Y, dist)
    # model.module.setTrainValidateMode(False)
    model.setValidateMode(False)

    y_pred = [0 if x <= normal_dist else 1 for x in max_dist]
    return AUC, PR, y_pred


def shuffle(X, Y, S):
    """
    Shuffle the datasets
    Args:
        X: input data
        Y: outlier labels
        S: sensitive attribute subgroups
    Returns: shuffled sets
    """
    set_seed(0)
    random_index = np.random.permutation(X.shape[0])
    return X[random_index], Y[random_index], S[random_index]


def main():
    parser = config_utils.get_args()
    args = parser.parse_args()
    options = vars(args)

    # Reproducibility
    set_seed(options['seed'])

    scaler = StandardScaler()

    # Get dataset
    print('Starting generate synthetic data')
    dataset = synthetic_dataset.SyntheticDataset(options['n'] * 200, options['d'], options['graph_type'],
                                                 options['degree'], options['sem_type'],
                                                 options['noise_scale'], options['dataset_type'], options['x_dim'],
                                                 options['alpha_cos'])
    print('Finish generating synthetic data')
    print('Starting split synthetic data')
    df_train, df_eval, df_test, df_eval_set, df_train_cf, df_eval_cf, df_test_cf, df_eval_set_cf = utils.get_samples(
        dataset)

    with_weight = 'true'
    if with_weight == 'true':
        weight = True
    else:
        weight = False

    # -----load sensitive subgroups----- #
    sensitive_attribute_group = df_train.iloc[:,0].values
    input = np.reshape(sensitive_attribute_group, (-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(input)
    one_hot = enc.transform(input).toarray()
    sensitive_attribute_group = np.argmax(one_hot, axis=1)

    # -----load dataset----- #
    X_norm = scaler.fit_transform(df_train.iloc[:, 1:-3].values)
    Y = df_train['label'].values
    X_norm, Y, sensitive_attribute_group = shuffle(X_norm, Y, sensitive_attribute_group)

    num_centroid = 1
    feature_dimension = X_norm.shape[1]
    embedded_dimension = 64
    num_subgroups = len(set(sensitive_attribute_group))
    # configuration = 90, 64 if X_norm.shape[0] < 10000 else 40, 256
    configuration = 90, 64 if X_norm.shape[0] < 10000 else 40, 256

    model = DCFOD(feature_dimension, num_centroid, embedded_dimension, num_subgroups, cuda)
    normal_dist = Train(model, X_norm, Y, sensitive_attribute_group, configuration[0], configuration[1], with_weight=False)

    test_X = scaler.transform(df_test.iloc[:, 1:-3].values.astype(np.float32))
    test_Y = df_test['label'].values
    _, _, unf_pred = validate(model, test_X, test_Y, 0.55)
    test_X_cf = scaler.transform(df_test_cf.iloc[:, 1:-3].values.astype(np.float32))
    test_Y_cf = df_test_cf['label'].values
    unf_AUC, unf_PR, unf_pred_cf = validate(model, test_X_cf, test_Y_cf, 0.55)
    print('Before fairness training results:')
    print(f'AUC value: {unf_AUC}')
    print(f'PR: {unf_PR}')
    print(classification_report(test_Y_cf, unf_pred_cf))
    print(confusion_matrix(test_Y_cf, unf_pred_cf))
    df_org = pd.DataFrame()
    df_org['label'] = test_Y
    df_org['pred'] = unf_pred
    df_org['pred_cf'] = unf_pred_cf

    print('Start fairness training')
    model = DCFOD(feature_dimension, num_centroid, embedded_dimension, num_subgroups, cuda)
    normal_dist = Train(model, X_norm, Y, sensitive_attribute_group, configuration[0], configuration[1],
                        with_weight=True)
    _, _, f_pred = validate(model, test_X, test_Y, 0.55)
    f_AUC, f_PR, f_pred_cf = validate(model, test_X_cf, test_Y_cf, 0.55)
    print('Before fairness training results:')
    print(f'AUC value: {f_AUC}')
    print(f'PR: {f_PR}')
    print(classification_report(test_Y_cf, f_pred_cf))
    print(confusion_matrix(test_Y_cf, f_pred_cf))
    df_ad = pd.DataFrame()
    df_ad['label'] = test_Y
    df_ad['pred'] = f_pred
    df_ad['pred_cf'] = f_pred_cf
    total = len(df_org)
    df_org['cf_changed'] = df_org['pred_cf'] - df_org['pred']
    df_ad['cf_changed'] = df_ad['pred_cf'] - df_ad['pred']
    df_org_cf = df_org.groupby(['cf_changed']).count().reset_index(drop=False)
    before_cf = sum(df_org_cf.loc[df_org_cf['cf_changed'] != 0]['label'].values)
    df_ad_cf = df_ad.groupby(['cf_changed']).count().reset_index(drop=False)
    after_cf = sum(df_ad_cf.loc[df_ad_cf['cf_changed'] != 0]['label'].values)
    print('Results for CF samples')
    print(f'Without fair, the prediction changed: {before_cf / total}')
    print(f'With fair, the prediction changed: {after_cf / total}')



if __name__ == '__main__':
    main()