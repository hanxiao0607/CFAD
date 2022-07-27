import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter


def set_seed(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_samples(dataset, upper=None, lower=None):
    df_X = pd.DataFrame(dataset.X.reshape(-1, 20))
    df_X['y'] = dataset.y
    df_X_cf = pd.DataFrame(dataset.X_cf.reshape(-1, 20))
    df_X_cf['y'] = dataset.y_cf
    # lst_y = df_X['y'].values
    lst_y = np.array(np.concatenate((df_X['y'].values, df_X_cf['y'].values)))
    if (upper is None) or (lower is None):
        ab_upper = np.quantile(lst_y, 0.99)
        ab_cf_uupper = np.quantile(lst_y, 0.90)
        ab_cf_ulower = np.quantile(lst_y, 0.85)
        # ab_cf_lupper = np.quantile(lst_y, 0.05)
        # ab_cf_llower = np.quantile(lst_y, 0.04)
        n_cf_upper = np.quantile(lst_y, 0.7)
        n_cf_lower = np.quantile(lst_y, 0.68)
        ab_lower = np.quantile(lst_y, 0.01)
        n_lower = np.quantile(lst_y, 0.3)
        n_upper = np.quantile(lst_y, 0.4)
    lst_n_c, lst_n_nc, lst_n_m, lst_ab_c, lst_ab_nc, lst_ab_m = [], [], [], [], [], []
    lst_n_c_cf, lst_n_nc_cf, lst_n_m_cf, lst_ab_c_cf, lst_ab_nc_cf, lst_ab_m_cf = [], [], [], [], [], []

    df_n = df_X.loc[(df_X['y'] > n_lower) & (df_X['y'] < n_upper)]
    df_n_cf = df_X_cf.loc[df_n.index]

    # df_n_c_cf = df_n_cf.loc[((df_n_cf['y'] > ab_cf_ulower) & (df_n_cf['y'] < ab_cf_uupper)) | ((df_n_cf['y'] > ab_cf_llower) & (df_n_cf['y'] < ab_cf_lupper))]
    df_n_c_cf = df_n_cf.loc[(df_n_cf['y'] > ab_cf_ulower) & (df_n_cf['y'] < ab_cf_uupper)]
    df_n_c = df_n.loc[df_n_c_cf.index]
    lst_n_c.extend(df_n_c.values)
    lst_n_c_cf.extend(df_n_c_cf.values)

    df_n_nc_cf = df_n_cf.loc[(df_n_cf['y'] > n_lower) & (df_n_cf['y'] < n_upper)]
    df_n_nc = df_n.loc[df_n_nc_cf.index]
    lst_n_nc.extend(df_n_nc.values)
    lst_n_nc_cf.extend(df_n_nc_cf.values)

    df_ab = df_X.loc[(df_X['y'] < ab_lower) | (df_X['y'] > ab_upper)]
    df_ab_cf = df_X_cf.loc[df_ab.index]

    df_ab_nc_cf = df_ab_cf.loc[(df_ab_cf['y'] < ab_lower) | (df_ab_cf['y'] > ab_upper)]
    df_ab_nc = df_ab.loc[df_ab_nc_cf.index]
    lst_ab_nc.extend(df_ab_nc.values)
    lst_ab_nc_cf.extend(df_ab_nc_cf.values)

    df_ab_c_cf = df_ab_cf.loc[(df_ab_cf['y'] > n_cf_lower) & (df_ab_cf['y'] < n_cf_upper)]
    df_ab_c = df_ab.loc[df_ab_c_cf.index]
    lst_ab_c.extend(df_ab_c.values)
    lst_ab_c_cf.extend(df_ab_c_cf.values)

    lst_temp = lst_n_c.copy()
    lst_temp.extend(lst_n_nc)

    #     lst_temp.extend(lst_n_m)
    lst_changed = [1 for _ in range(len(lst_n_c))]
    lst_changed.extend([0 for _ in range(len(lst_n_nc))])
    #     lst_changed.extend([-1 for _ in range(len(lst_n_m))])
    df_n = pd.DataFrame(lst_temp)
    df_n['changed'] = lst_changed
    df_n['label'] = 0
    df_n = df_n.sample(n=len(df_n), random_state=42)

    lst_temp = lst_ab_c.copy()
    lst_temp.extend(lst_ab_nc)
    #     lst_temp.extend(lst_ab_m)
    lst_changed = [1 for _ in range(len(lst_ab_c))]
    lst_changed.extend([0 for _ in range(len(lst_ab_nc))])
    #     lst_changed.extend([-1 for _ in range(len(lst_ab_m))])
    df_ab = pd.DataFrame(lst_temp)
    df_ab['changed'] = lst_changed
    df_ab['label'] = 1

    lst_temp = lst_n_c_cf.copy()
    lst_temp.extend(lst_n_nc_cf)
    #     lst_temp.extend(lst_n_m_cf)
    lst_changed = [1 for _ in range(len(lst_n_c_cf))]
    lst_changed.extend([0 for _ in range(len(lst_n_nc_cf))])
    #     lst_changed.extend([-1 for _ in range(len(lst_n_m_cf))])
    lst_label = [1 for _ in range(len(lst_n_c_cf))]
    lst_label.extend([0 for _ in range(len(lst_n_nc_cf))])
    #     lst_label.extend([0 for _ in range(len(lst_n_m_cf))])
    df_n_cf = pd.DataFrame(lst_temp)
    df_n_cf['changed'] = lst_changed
    df_n_cf['label'] = lst_label
    df_n_cf = df_n_cf.loc[df_n.index]

    lst_temp = lst_ab_c_cf.copy()
    lst_temp.extend(lst_ab_nc_cf)
    #     lst_temp.extend(lst_ab_m_cf)
    lst_changed = [1 for _ in range(len(lst_ab_c_cf))]
    lst_changed.extend([0 for _ in range(len(lst_ab_nc_cf))])
    #     lst_changed.extend([-1 for _ in range(len(lst_ab_m_cf))])
    lst_label = [0 for _ in range(len(lst_ab_c_cf))]
    lst_label.extend([1 for _ in range(len(lst_ab_nc_cf))])
    #     lst_label.extend([0 for _ in range(len(lst_ab_m_cf))])
    df_ab_cf = pd.DataFrame(lst_temp)
    df_ab_cf['changed'] = lst_changed
    df_ab_cf['label'] = lst_label

    major = len(df_n.loc[df_n[0] == 1])
    minor = int(major * 0.1)
    df_n_nc_major = df_n.loc[(df_n[0] == 1) & (df_n['changed'] == 0)]
    df_n_c_major = df_n.loc[(df_n[0] == 1) & (df_n['changed'] == 1)]
    #     df_n_m_major = df_n.loc[(df_n[0]==1) & (df_n['changed']==-1)]
    df_n_nc_minor = df_n.loc[(df_n[0] == -1) & (df_n['changed'] == 0)]
    df_n_c_minor = df_n.loc[(df_n[0] == -1) & (df_n['changed'] == 1)]
    #     df_n_m_minor = df_n.loc[(df_n[0]==-1) & (df_n['changed']==-1)]
    print(f"Normal major nc:{len(df_n_nc_major)}, c:{len(df_n_c_major)}")
    print(f"Normal minor nc:{len(df_n_nc_minor)}, c:{len(df_n_c_minor)}")

    df_ab_nc_major = df_ab.loc[(df_ab[0] == 1) & (df_ab['changed'] == 0)]
    df_ab_c_major = df_ab.loc[(df_ab[0] == 1) & (df_ab['changed'] == 1)]
    #     df_ab_m_major = df_ab.loc[(df_ab[0]==1) & (df_ab['changed']==-1)]
    df_ab_nc_minor = df_ab.loc[(df_ab[0] == -1) & (df_ab['changed'] == 0)]
    df_ab_c_minor = df_ab.loc[(df_ab[0] == -1) & (df_ab['changed'] == 1)]
    #     df_ab_m_minor = df_ab.loc[(df_ab[0]==-1) & (df_ab['changed']==-1)]
    print(f"Abnormal major nc:{len(df_ab_nc_major)}, c:{len(df_ab_c_major)}")
    print(f"Abnormal minor nc:{len(df_ab_nc_minor)}, c:{len(df_ab_c_minor)}")

    n_train = 3000
    df_train = pd.concat([df_n_nc_major.iloc[:n_train], df_n_c_major.iloc[:n_train], \
                          df_n_nc_minor.iloc[:n_train], df_n_c_minor.iloc[:n_train]])
    df_train = df_train.sample(n=len(df_train), random_state=42)
    df_eval_set = pd.concat([df_n_nc_major.iloc[0:1], df_n_c_major.iloc[0:1], \
                             df_n_nc_minor.iloc[0:1], df_n_c_minor.iloc[0:1]])
    df_test_n = pd.concat([df_n_nc_major.iloc[n_train:n_train+1000], df_n_c_major.iloc[n_train:n_train+1000], \
                           df_n_nc_minor.iloc[n_train:n_train+1000], df_n_c_minor.iloc[n_train:n_train+1000]])
    df_test_ab = pd.concat([df_ab_nc_major.iloc[:100], df_ab_c_major.iloc[:100], \
                            df_ab_nc_minor.iloc[:100], df_ab_c_minor.iloc[:100]])

    df_train_cf = df_n_cf.loc[df_train.index]
    df_eval_set_cf = df_n_cf.loc[df_eval_set.index]
    df_test_n_cf = df_n_cf.loc[df_test_n.index]
    df_test_ab_cf = df_ab_cf.loc[df_test_ab.index]

    df_eval = df_train.iloc[-1000:].reset_index(drop=True)
    df_train = df_train.iloc[:-1000].reset_index(drop=True)
    df_eval_set.reset_index(drop=True, inplace=True)
    df_test = pd.concat([df_test_n, df_test_ab]).reset_index(drop=True)

    df_eval_cf = df_train_cf.iloc[-1000:].reset_index(drop=True)
    df_train_cf = df_train_cf.iloc[:-1000].reset_index(drop=True)
    df_eval_set_cf.reset_index(drop=True, inplace=True)
    df_test_cf = pd.concat([df_test_n_cf, df_test_ab_cf]).reset_index(drop=True)
    try:
        df_train.to_csv('../data/train.csv')
        df_test.to_csv('../data/test.csv')
        df_test_cf.to_csv('../data/test_cf.csv')
    except:
        df_train.to_csv('data/train.csv')
        df_test.to_csv('data/test.csv')
        df_test_cf.to_csv('data/test_cf.csv')
    return df_train, df_eval, df_test, df_eval_set, df_train_cf, df_eval_cf, df_test_cf, df_eval_set_cf


class CFDataset(Dataset):
    def __init__(self, X, do):
        self.X = X
        self.do = do

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.X[idx], self.do[idx])


def pretrain_split(df_train, df_eval, scaler, adult=0):
    if not adult:
        train_X = scaler.fit_transform(df_train.iloc[:, 1:-3].values.astype(np.float32))
        lst_temp = [1 for _ in range(len(train_X))]

        train_iter = DataLoader(CFDataset(torch.tensor(train_X.astype(np.float32)), torch.LongTensor(lst_temp)),
                                batch_size=128, shuffle=True, worker_init_fn=np.random.seed(0))
        # print(len(train_X))
        eval_X = scaler.transform(df_eval.iloc[:, 1:-3].values.astype(np.float32))
        lst_temp = [1 for _ in range(len(eval_X))]

        eval_iter = DataLoader(CFDataset(torch.tensor(eval_X.astype(np.float32)), torch.LongTensor(lst_temp)),
                               batch_size=32, shuffle=False)

    else:
        train_X = df_train.iloc[:, 1:-1].values.astype(np.float32)
        lst_temp = [1 for _ in range(len(train_X))]

        train_iter = DataLoader(CFDataset(torch.tensor(train_X.astype(np.float32)), torch.LongTensor(lst_temp)),
                                batch_size=128, shuffle=False, worker_init_fn=np.random.seed(0))
        # print(len(train_X))
        eval_X = df_eval.iloc[:, 1:-1].values.astype(np.float32)
        lst_temp = [1 for _ in range(len(eval_X))]

        eval_iter = DataLoader(CFDataset(torch.tensor(eval_X.astype(np.float32)), torch.LongTensor(lst_temp)),
                               batch_size=32, shuffle=False)
    return train_iter, eval_iter, scaler

def get_pretrain_result(gaes, aae_trainer, df_test, df_test_cf=[], ratio=1, scaler=None, val=0):
    R_aae = aae_trainer.max_dist * ratio
    test_iter = DataLoader(scaler.transform(df_test.iloc[:, 1:-3].values.astype(np.float32)), batch_size=32, shuffle=False)
    lst_pred, lst_score = aae_trainer._evaluation(test_iter, df_test['label'], r=R_aae)
    lst_val = lst_score.copy()
    print('Original')
    print(classification_report(y_true=df_test['label'], y_pred=lst_pred, digits=5))
    print(confusion_matrix(y_true=df_test['label'], y_pred=lst_pred))
    print(f"AUC-PR: {average_precision_score(y_true=df_test['label'], y_score=lst_score)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=df_test['label'], y_score=lst_score)}")
    df_org = pd.DataFrame()
    df_org['label'] = df_test['label'].values
    df_org['pred'] = lst_pred

    # test_do = gaes.net.get_result(
    #     torch.Tensor(df_test.iloc[:, :-3].values.astype(np.float32).reshape(-1, 20, 1)).to(device),
    #     do=1).detach().cpu().numpy()
    # test_iter = DataLoader(test_do.reshape(-1, 20)[:, 1:], batch_size=32, shuffle=False)
    # lst_pred = aae_trainer._evaluation(test_iter, df_test_cf['label'], r=R_aae)
    # print('Do')
    # print(classification_report(y_true=df_test_cf['label'], y_pred=lst_pred, digits=5))
    # print(confusion_matrix(y_true=df_test_cf['label'], y_pred=lst_pred))
    # print(f"AUC-PR: {average_precision_score(y_true=df_test_cf['label'], y_score=lst_pred)}")
    # print(f"AUC-ROC: {roc_auc_score(y_true=df_test_cf['label'], y_score=lst_pred)}")
    # df_org['pred_do'] = lst_pred

    lst_val_cf = []
    if len(df_test_cf) >= 1:
        test_iter = DataLoader(scaler.transform(df_test_cf.iloc[:, 1:-3].values.astype(np.float32).reshape(-1, 19)), batch_size=32,
                               shuffle=False)
        lst_pred, lst_score  = aae_trainer._evaluation(test_iter, df_test_cf['label'], r=R_aae)
        lst_val_cf = lst_score.copy()
        print('CF')
        print(classification_report(y_true=df_test_cf['label'], y_pred=lst_pred, digits=5))
        print(confusion_matrix(y_true=df_test_cf['label'], y_pred=lst_pred))
        print(f"AUC-PR: {average_precision_score(y_true=df_test_cf['label'], y_score=lst_score)}")
        print(f"AUC-ROC: {roc_auc_score(y_true=df_test_cf['label'], y_score=lst_score)}")
        df_org['pred_cf'] = lst_pred
    if val == 0:
        return df_org
    else:
        return df_org, df_test, lst_val, lst_val_cf, R_aae


def get_pretrain_results_adult(aae_trainer, df_test, test_do, val=0):
    R_aae = aae_trainer.max_dist
    test_iter = DataLoader(df_test.iloc[:, 1:-1].values.astype(np.float32), batch_size=32,
                           shuffle=False)
    lst_pred, lst_score = aae_trainer._evaluation(test_iter, df_test['y'], r=R_aae)
    lst_val = lst_score.copy()
    print('Original')
    print(classification_report(y_true=df_test['y'], y_pred=lst_pred, digits=5))
    print(confusion_matrix(y_true=df_test['y'], y_pred=lst_pred))
    print(f"AUC-PR: {average_precision_score(y_true=df_test['y'], y_score=lst_score)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=df_test['y'], y_score=lst_score)}")
    df_org = pd.DataFrame()
    df_org['label'] = df_test['y'].values
    df_org['pred'] = lst_pred

    test_iter = DataLoader(test_do.astype(np.float32),batch_size=32,shuffle=False)
    lst_pred, lst_score = aae_trainer._evaluation(test_iter, df_test['y'], r=R_aae)
    lst_val_cf = lst_score.copy()
    print('Generated CF')
    print(classification_report(y_true=df_test['y'], y_pred=lst_pred, digits=5))
    print(confusion_matrix(y_true=df_test['y'], y_pred=lst_pred))
    print(f"AUC-PR: {average_precision_score(y_true=df_test['y'], y_score=lst_score)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=df_test['y'], y_score=lst_score)}")
    df_org['pred_do'] = lst_pred
    if val == 0:
        return df_org
    else:
        return df_org, df_test, lst_val, lst_val_cf, R_aae


def retrain_split(gaes, df_train, df_eval, scaler, device='cuda:0', adult=0):
    if not adult:
        X_do = gaes.net.get_result(torch.Tensor(df_train.iloc[:, :-3].values.astype(np.float32).reshape(-1, 20, 1)).to(device),
                                   do=1).detach().cpu().numpy().reshape(-1, 20)[:, 1:]
        # X_do = np.delete(X_do, df_train_cf.loc[df_train_cf['label'] == 1].index, axis=0)
        train_X_do = X_do
        train_X_or = df_train.iloc[:, 1:-3].values.astype(np.float32)
        lst_temp = [1 for _ in range(len(train_X_do))]
        lst_temp.extend([0 for _ in range(len(train_X_or))])
        train_X = scaler.transform(np.concatenate((train_X_do, train_X_or), axis=0))

        train_iter = DataLoader(CFDataset(torch.tensor(train_X.astype(np.float32)), torch.LongTensor(lst_temp)),
                                batch_size=128, shuffle=True, worker_init_fn=np.random.seed(0))
        print(len(train_X))
        eval_X_do = gaes.net.get_result(
            torch.Tensor(df_eval.iloc[:, :-3].values.astype(np.float32).reshape(-1, 20, 1)).to(device),
            do=1).detach().cpu().numpy().reshape(-1, 20)[:, 1:]
        eval_X_or = df_eval.iloc[:, 1:-3].values.astype(np.float32)
        lst_temp = [1 for _ in range(len(eval_X_do))]
        lst_temp.extend([0 for _ in range(len(eval_X_or))])
        eval_X = scaler.transform(np.concatenate((eval_X_do, eval_X_or), axis=0))

        eval_iter = DataLoader(CFDataset(torch.tensor(eval_X.astype(np.float32)), torch.LongTensor(lst_temp)),
                               batch_size=32, shuffle=False)
    else:
        X_do = gaes.net.get_result(
            torch.Tensor(df_train.iloc[:, :-1].values.astype(np.float32).reshape(len(df_train), -1, 1)).to(device),
            do=1).detach().cpu().numpy().reshape(len(df_train), -1)[:, 1:]
        # X_do = np.delete(X_do, df_train_cf.loc[df_train_cf['label'] == 1].index, axis=0)
        train_X_do = X_do
        train_X_or = df_train.iloc[:, 1:-1].values.astype(np.float32)
        lst_temp = [1 for _ in range(len(train_X_do))]
        lst_temp.extend([0 for _ in range(len(train_X_or))])
        train_X = np.concatenate((train_X_do, train_X_or), axis=0)

        train_iter = DataLoader(CFDataset(torch.tensor(train_X.astype(np.float32)), torch.LongTensor(lst_temp)),
                                batch_size=128, shuffle=True, worker_init_fn=np.random.seed(0))
        print(len(train_X))
        eval_X_do = gaes.net.get_result(
            torch.Tensor(df_eval.iloc[:, :-1].values.astype(np.float32).reshape(len(df_eval), -1, 1)).to(device),
            do=1).detach().cpu().numpy().reshape(len(df_eval), -1)[:, 1:]
        eval_X_or = df_eval.iloc[:, 1:-1].values.astype(np.float32)
        lst_temp = [1 for _ in range(len(eval_X_do))]
        lst_temp.extend([0 for _ in range(len(eval_X_or))])
        eval_X = np.concatenate((eval_X_do, eval_X_or), axis=0)

        eval_iter = DataLoader(CFDataset(torch.tensor(eval_X.astype(np.float32)), torch.LongTensor(lst_temp)),
                               batch_size=32, shuffle=False)

    return train_iter, eval_iter

def get_retrain_result(gaes, aae_trainer, df_test, df_test_cf=[], ratio=1, scaler=None, val=0):
    R_aae = aae_trainer.max_dist * ratio
    test_iter = DataLoader(scaler.transform(df_test.iloc[:, 1:-3].values.astype(np.float32)), batch_size=32, shuffle=False)
    lst_pred, lst_score = aae_trainer._evaluation(test_iter, df_test['label'], r=R_aae)
    lst_val = lst_score.copy()
    print('Original')
    print(classification_report(y_true=df_test['label'], y_pred=lst_pred, digits=5))
    print(confusion_matrix(y_true=df_test['label'], y_pred=lst_pred))
    print(f"AUC-PR: {average_precision_score(y_true=df_test['label'], y_score=lst_score)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=df_test['label'], y_score=lst_score)}")
    df_ad = pd.DataFrame()
    df_ad['label'] = df_test['label'].values
    df_ad['pred'] = lst_pred

    # test_do = gaes.net.get_result(
    #     torch.Tensor(df_test.iloc[:, :-3].values.astype(np.float32).reshape(-1, 20, 1)).cuda(),
    #     do=1).detach().cpu().numpy()
    # test_iter = DataLoader(test_do.reshape(-1, 20)[:, 1:], batch_size=32, shuffle=False)
    # lst_pred = aae_trainer._evaluation(test_iter, df_test_cf['label'], r=R_aae)
    # print('Do')
    # print(classification_report(y_true=df_test_cf['label'], y_pred=lst_pred, digits=5))
    # print(confusion_matrix(y_true=df_test_cf['label'], y_pred=lst_pred))
    # print(f"AUC-PR: {average_precision_score(y_true=df_test_cf['label'], y_score=lst_pred)}")
    # print(f"AUC-ROC: {roc_auc_score(y_true=df_test_cf['label'], y_score=lst_pred)}")
    # df_ad['pred_do'] = lst_pred
    lst_val_cf = []
    if len(df_test_cf) >= 1:
        test_iter = DataLoader(scaler.transform(df_test_cf.iloc[:, 1:-3].values.astype(np.float32).reshape(-1, 19)), batch_size=32,
                               shuffle=False)
        lst_pred, lst_score = aae_trainer._evaluation(test_iter, df_test_cf['label'], r=R_aae)
        lst_val_cf = lst_score.copy()
        print('CF')
        print(classification_report(y_true=df_test_cf['label'], y_pred=lst_pred, digits=5))
        print(confusion_matrix(y_true=df_test_cf['label'], y_pred=lst_pred))
        print(f"AUC-PR: {average_precision_score(y_true=df_test_cf['label'], y_score=lst_score)}")
        print(f"AUC-ROC: {roc_auc_score(y_true=df_test_cf['label'], y_score=lst_score)}")
        df_ad['pred_cf'] = lst_pred
    if val == 0:
        return df_ad
    else:
        return df_ad, df_test, lst_val, lst_val_cf, R_aae


def get_retrain_results_adult(aae_trainer, df_test, test_do, val=0):
    R_aae = aae_trainer.max_dist
    test_iter = DataLoader(df_test.iloc[:, 1:-1].values.astype(np.float32), batch_size=32,
                           shuffle=False)
    lst_pred, lst_score = aae_trainer._evaluation(test_iter, df_test['y'], r=R_aae)
    lst_val = lst_score.copy()
    print('Original')
    print(classification_report(y_true=df_test['y'], y_pred=lst_pred, digits=5))
    print(confusion_matrix(y_true=df_test['y'], y_pred=lst_pred))
    print(f"AUC-PR: {average_precision_score(y_true=df_test['y'], y_score=lst_score)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=df_test['y'], y_score=lst_score)}")
    df_ad = pd.DataFrame()
    df_ad['label'] = df_test['y'].values
    df_ad['pred'] = lst_pred

    test_iter = DataLoader(test_do.astype(np.float32), batch_size=32, shuffle=False)
    lst_pred, lst_score = aae_trainer._evaluation(test_iter, df_test['y'], r=R_aae)
    lst_val_cf = lst_score.copy()
    print('Generated CF')
    print(classification_report(y_true=df_test['y'], y_pred=lst_pred, digits=5))
    print(confusion_matrix(y_true=df_test['y'], y_pred=lst_pred))
    print(f"AUC-PR: {average_precision_score(y_true=df_test['y'], y_score=lst_score)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=df_test['y'], y_score=lst_score)}")
    df_ad['pred_do'] = lst_pred
    if val == 0:
        return df_ad
    else:
        return df_ad, df_test, lst_val, lst_val_cf, R_aae


def get_fairness_result(df_org, df_ad, cf=0):
    # df_org['do_changed'] = df_org['pred_do'] - df_org['pred']
    # df_ad['do_changed'] = df_ad['pred_do'] - df_ad['pred']
    assert len(df_org) == len(df_ad), 'Length should be the same!'
    total = len(df_org)
    # df_org_do = df_org.groupby(['do_changed']).count().reset_index(drop=False)
    # before_do = sum(df_org_do.loc[df_org_do['do_changed'] != 0]['label'].values)
    # df_ad_do = df_ad.groupby(['do_changed']).count().reset_index(drop=False)
    # after_do = sum(df_ad_do.loc[df_ad_do['do_changed'] != 0]['label'].values)
    # print('Results for DO samples')
    # print(f'Without fair, the prediction changed: {before_do/total}')
    # print(f'With fair, the prediction changed: {after_do/total}')


    if cf:
        df_org['cf_changed'] = df_org['pred_cf'] - df_org['pred']
        df_ad['cf_changed'] = df_ad['pred_cf'] - df_ad['pred']
        df_org_cf = df_org.groupby(['cf_changed']).count().reset_index(drop=False)
        before_cf = sum(df_org_cf.loc[df_org_cf['cf_changed'] != 0]['label'].values)
        df_ad_cf = df_ad.groupby(['cf_changed']).count().reset_index(drop=False)
        after_cf = sum(df_ad_cf.loc[df_ad_cf['cf_changed'] != 0]['label'].values)
        print('Results for CF samples')
        print(f'Without fair, the prediction changed: {before_cf / total}')
        print(f'With fair, the prediction changed: {after_cf / total}')
    else:
        df_org['do_changed'] = df_org['pred_do'] - df_org['pred']
        df_ad['do_changed'] = df_ad['pred_do'] - df_ad['pred']
        df_org_cf = df_org.groupby(['do_changed']).count().reset_index(drop=False)
        before_cf = sum(df_org_cf.loc[df_org_cf['do_changed'] != 0]['label'].values)
        df_ad_cf = df_ad.groupby(['do_changed']).count().reset_index(drop=False)
        after_cf = sum(df_ad_cf.loc[df_ad_cf['do_changed'] != 0]['label'].values)
        print('Results for Generated CF samples')
        print(f'Without fair, the prediction changed: {before_cf / total}')
        print(f'With fair, the prediction changed: {after_cf / total}')

def load_data(flag=0):
    if flag == 0:
        df_train = pd.read_csv('../data/train.csv', index_col=0)
        df_test = pd.read_csv('../data/test.csv', index_col=0)
        df_test_cf = pd.read_csv('../data/test_cf.csv', index_col=0)
    elif flag == 1:
        df_train = pd.read_csv('../data/adult_train.csv', index_col=0)
        df_test = pd.read_csv('../data/adult_test.csv', index_col=0)
        df_test_cf = pd.read_csv('../data/adult_do.csv', index_col=0)
    elif flag == 2:
        df_train = pd.read_csv('../data/compas_train.csv', index_col=0)
        df_test = pd.read_csv('../data/compas_test.csv', index_col=0)
        df_test_cf = pd.read_csv('../data/compas_do.csv', index_col=0)
    else:
        print('Please given a number from 0 to 2. 0 for synthetic, 1 for adult, and 2 for compas.')
    return df_train, df_test, df_test_cf

def adult_preprocessing(dir='data/adult.data', n_train=10000, n_test=2000):
    df_data = pd.read_csv(dir, header=None, names=['age', 'workclass', 'fnlwgt', 'education',
                                                                    'education-num', 'marital-status', 'occupation',
                                                                    'relationship', 'race', 'sex', 'capital-gain',
                                                                    'capital-loss',
                                                                    'hours-per-week', 'native-country', 'y'])
    for i in range(len(df_data.columns)):
        most_frequent = df_data.iloc[:, i].value_counts()[:1].index.tolist()[0]
        for j in range(len(df_data.iloc[:, i])):
            if df_data.iloc[j, i] == '?':
                df_data.iloc[j, i] = most_frequent
    df_data.loc[df_data['y'] == ' >50K', 'y'] = 1
    df_data.loc[df_data['y'] == ' <=50K', 'y'] = 0
    df_data.loc[df_data['sex'] == ' Female', 'sex'] = 1
    df_data.loc[df_data['sex'] == ' Male', 'sex'] = -1
    for i in ['workclass', 'marital-status', 'occupation', 'education', 'relationship', 'race', 'native-country', 'marital-status']:
        data = Counter(df_data[i].values)
        val = data.most_common(1)[0][0]
        df_data.loc[df_data[i] != val, i] = 0
        df_data.loc[df_data[i] == val, i] = 1
    df_data = df_data[['sex', 'age', 'native-country', 'race', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                       'relationship', 'capital-gain', 'capital-loss',
                       'hours-per-week', 'y']]
    scaler = MinMaxScaler((-3,3))
    # scaler = MinMaxScaler()
    df_data = df_data.sample(n=len(df_data), random_state=42)
    df_data['y'] = df_data['y'].astype(int)
    df_n = df_data.loc[df_data['y'] == 0].copy()
    df_n.iloc[:, 1:-1] = scaler.fit_transform(df_n.iloc[:, 1:-1].values)
    df_ab = df_data.loc[df_data['y'] == 1].copy()
    df_ab.iloc[:, 1:-1] = scaler.transform(df_ab.iloc[:, 1:-1].values)
    df_train = df_n.iloc[:n_train]
    # df_test = pd.concat([df_n.iloc[n_train:n_train+4000], df_ab.iloc[:800]])
    df_test = pd.concat([df_n.iloc[n_train:n_train+n_test], df_ab.iloc[:int(0.2*n_test)]])
    return df_train, df_test


def compas_preprocessing(dir='data/compas-scores-two-years.txt', n_train=2000, n_test=2000):
    df = pd.read_csv('data/compas-scores-two-years.txt', usecols=['race', 'sex', 'age', 'juv_fel_count', 'decile_score', \
                                                                  'juv_misd_count', 'juv_other_count', 'priors_count', \
                                                                  'score_text', 'two_year_recid'])
    df_sel = df.loc[df['race'].isin(['African-American', 'Caucasian'])]
    df_sel = df_sel[['race', 'sex', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', \
                     'decile_score', 'two_year_recid']]
    # juvenile_misdemeanors_count; felony
    df_sel.loc[df_sel['race'] == 'African-American', 'race'] = -1
    df_sel.loc[df_sel['race'] == 'Caucasian', 'race'] = 1
    df_sel.loc[df_sel['sex'] == 'Male', 'sex'] = 0
    df_sel.loc[df_sel['sex'] == 'Female', 'sex'] = 1
    df_sel.rename(columns={'two_year_recid': 'y'}, inplace=True)
    df_sel.reset_index(drop=True, inplace=True)

    # scaler = MinMaxScaler((-3, 3))
    scaler = MinMaxScaler()
    df_n = df_sel.loc[df_sel['y'] == 0].copy()
    df_n.iloc[:,1:-1] = scaler.fit_transform(df_n.iloc[:,1:-1].values)
    df_n = df_n.sample(n=len(df_n), random_state=42)
    df_ab = df_sel.loc[df_sel['y'] == 1].copy()
    df_ab = df_ab.sample(n=len(df_ab), random_state=42)
    df_ab.iloc[:, 1:-1] = scaler.transform(df_ab.iloc[:, 1:-1].values)
    df_train = df_n.iloc[:n_train].copy()
    df_test = pd.concat([df_n.iloc[n_train:], df_ab.iloc[:int(0.3*len(df_n.iloc[n_train:]))]])
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    return df_train, df_test