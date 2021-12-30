import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler


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

    if (upper is None) or (lower is None):
        upper = df_X.loc[df_X[0] == 1]['y'].quantile(0.99)
        inner_upper = df_X.loc[df_X[0] == 1]['y'].quantile(0.75)
        lower = df_X.loc[df_X[0] == 1]['y'].quantile(0.01)
        inner_lower = df_X.loc[df_X[0] == 1]['y'].quantile(0.25)
    lst_n_c, lst_n_nc, lst_n_m, lst_ab_c, lst_ab_nc, lst_ab_m = [], [], [], [], [], []
    lst_n_c_cf, lst_n_nc_cf, lst_n_m_cf, lst_ab_c_cf, lst_ab_nc_cf, lst_ab_m_cf = [], [], [], [], [], []
    for i in range(len(df_X)):
        sample_X = df_X.iloc[i]
        sample_X_cf = df_X_cf.iloc[i]
        if sample_X['y'] > inner_lower and sample_X['y'] < inner_upper:
            if (sample_X_cf['y'] > upper) or (sample_X_cf['y'] < lower):
                lst_n_c.append(sample_X.values)
                lst_n_c_cf.append(sample_X_cf.values)
            elif sample_X_cf['y'] > inner_lower and sample_X_cf['y'] < inner_upper:
                lst_n_nc.append(sample_X.values)
                lst_n_nc_cf.append(sample_X_cf.values)
            else:
                lst_n_m.append(sample_X.values)
                lst_n_m_cf.append(sample_X_cf.values)

        elif (sample_X['y'] < lower) or (sample_X['y'] > upper):
            if (sample_X_cf['y'] > upper) or (sample_X_cf['y'] < lower):
                lst_ab_nc.append(sample_X.values)
                lst_ab_nc_cf.append(sample_X_cf.values)
            elif sample_X_cf['y'] > inner_lower and sample_X_cf['y'] < inner_upper:
                lst_ab_c.append(sample_X.values)
                lst_ab_c_cf.append(sample_X_cf.values)
            else:
                lst_ab_m.append(sample_X.values)
                lst_ab_m_cf.append(sample_X_cf.values)

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

    #     df_train = pd.concat([df_n_nc_major.iloc[:1000], df_n_c_major.iloc[:1], df_n_m_major.iloc[:10000],\
    #                           df_n_nc_minor.iloc[:100], df_n_c_minor.iloc[:1], df_n_m_minor.iloc[:1000]])
    #     df_train = df_train.sample(n=len(df_train), random_state=42)
    #     df_eval_set = pd.concat([df_n_nc_major.iloc[1000:1100], df_n_c_major.iloc[1000:1100], df_n_m_major.iloc[10000:11000],\
    #                              df_n_nc_minor.iloc[100:110], df_n_c_minor.iloc[100:110], df_n_m_minor.iloc[1000:1100]])
    #     df_test_n = pd.concat([df_n_nc_major.iloc[1100:2100], df_n_c_major.iloc[1100:2100], df_n_m_major.iloc[11000:21000],\
    #                          df_n_nc_minor.iloc[110:210], df_n_c_minor.iloc[110:210], df_n_m_minor.iloc[1100:2100]])
    #     df_test_ab = pd.concat([df_ab_nc_major.iloc[:10], df_ab_c_major.iloc[:10], df_ab_m_major.iloc[:100],\
    #                          df_ab_nc_minor.iloc[:100], df_ab_c_minor.iloc[:100], df_ab_m_minor.iloc[:1000]])

    df_train = pd.concat([df_n_nc_major.iloc[:10000], df_n_c_major.iloc[:1000], \
                          df_n_nc_minor.iloc[:1000], df_n_c_minor.iloc[:100]])
    df_train = df_train.sample(n=len(df_train), random_state=42)
    df_eval_set = pd.concat([df_n_nc_major.iloc[0:1], df_n_c_major.iloc[0:1], \
                             df_n_nc_minor.iloc[0:1], df_n_c_minor.iloc[0:1]])
    # df_test_n = pd.concat([df_n_nc_major.iloc[10000:11000], df_n_c_major.iloc[1000:2000], \
    #                        df_n_nc_minor.iloc[1000:2000], df_n_c_minor.iloc[100:1100]])
    # df_test_ab = pd.concat([df_ab_nc_major.iloc[:100], df_ab_c_major.iloc[:100], \
    #                         df_ab_nc_minor.iloc[:100], df_ab_c_minor.iloc[:100]])
    df_test_n = pd.concat([df_n_nc_major.iloc[10000:10000], df_n_c_major.iloc[1000:2000], \
                           df_n_nc_minor.iloc[1000:1000], df_n_c_minor.iloc[100:1100]])
    df_test_ab = pd.concat([df_ab_nc_major.iloc[:0], df_ab_c_major.iloc[:100], \
                            df_ab_nc_minor.iloc[:0], df_ab_c_minor.iloc[:100]])

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


def pretrain_split(df_train, df_eval, scaler):
    train_X = scaler.fit_transform(df_train.iloc[:, 1:-3].values.astype(np.float32))
    lst_temp = [1 for _ in range(len(train_X))]

    train_iter = DataLoader(CFDataset(torch.tensor(train_X.astype(np.float32)), torch.LongTensor(lst_temp)),
                            batch_size=128, shuffle=True, worker_init_fn=np.random.seed(0))
    # print(len(train_X))
    eval_X = scaler.transform(df_eval.iloc[:, 1:-3].values.astype(np.float32))
    lst_temp = [1 for _ in range(len(eval_X))]

    eval_iter = DataLoader(CFDataset(torch.tensor(eval_X.astype(np.float32)), torch.LongTensor(lst_temp)),
                           batch_size=32, shuffle=False)
    return train_iter, eval_iter, scaler

def get_pretrain_result(gaes, aae_trainer, df_test, df_test_cf=[], ratio=1, scaler=None):
    R_aae = aae_trainer.max_dist * ratio
    test_iter = DataLoader(scaler.transform(df_test.iloc[:, 1:-3].values.astype(np.float32)), batch_size=32, shuffle=False)
    lst_pred, lst_score = aae_trainer._evaluation(test_iter, df_test['label'], r=R_aae)
    print('Original')
    print(classification_report(y_true=df_test['label'], y_pred=lst_pred, digits=5))
    print(confusion_matrix(y_true=df_test['label'], y_pred=lst_pred))
    print(f"AUC-PR: {average_precision_score(y_true=df_test['label'], y_score=lst_score)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=df_test['label'], y_score=lst_score)}")
    df_org = pd.DataFrame()
    df_org['label'] = df_test['label'].values
    df_org['pred'] = lst_pred

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
    # df_org['pred_do'] = lst_pred


    if len(df_test_cf) >= 1:
        test_iter = DataLoader(scaler.transform(df_test_cf.iloc[:, 1:-3].values.astype(np.float32).reshape(-1, 19)), batch_size=32,
                               shuffle=False)
        lst_pred, lst_score  = aae_trainer._evaluation(test_iter, df_test_cf['label'], r=R_aae)
        print('CF')
        print(classification_report(y_true=df_test_cf['label'], y_pred=lst_pred, digits=5))
        print(confusion_matrix(y_true=df_test_cf['label'], y_pred=lst_pred))
        print(f"AUC-PR: {average_precision_score(y_true=df_test_cf['label'], y_score=lst_score)}")
        print(f"AUC-ROC: {roc_auc_score(y_true=df_test_cf['label'], y_score=lst_score)}")
        df_org['pred_cf'] = lst_pred

    return df_org

def get_pretrain_result(gaes, aae_trainer, df_test, df_test_cf=[], ratio=1, scaler=None):
    R_aae = aae_trainer.max_dist * ratio
    test_iter = DataLoader(scaler.transform(df_test.iloc[:, 1:-3].values.astype(np.float32)), batch_size=32, shuffle=False)
    lst_pred, lst_score = aae_trainer._evaluation(test_iter, df_test['label'], r=R_aae)
    print('Original')
    print(classification_report(y_true=df_test['label'], y_pred=lst_pred, digits=5))
    print(confusion_matrix(y_true=df_test['label'], y_pred=lst_pred))
    print(f"AUC-PR: {average_precision_score(y_true=df_test['label'], y_score=lst_score)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=df_test['label'], y_score=lst_score)}")
    df_org = pd.DataFrame()
    df_org['label'] = df_test['label'].values
    df_org['pred'] = lst_pred

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
    # df_org['pred_do'] = lst_pred


    if len(df_test_cf) >= 1:
        test_iter = DataLoader(scaler.transform(df_test_cf.iloc[:, 1:-3].values.astype(np.float32).reshape(-1, 19)), batch_size=32,
                               shuffle=False)
        lst_pred, lst_score  = aae_trainer._evaluation(test_iter, df_test_cf['label'], r=R_aae)
        print('CF')
        print(classification_report(y_true=df_test_cf['label'], y_pred=lst_pred, digits=5))
        print(confusion_matrix(y_true=df_test_cf['label'], y_pred=lst_pred))
        print(f"AUC-PR: {average_precision_score(y_true=df_test_cf['label'], y_score=lst_score)}")
        print(f"AUC-ROC: {roc_auc_score(y_true=df_test_cf['label'], y_score=lst_score)}")
        df_org['pred_cf'] = lst_pred

    return df_org

def retrain_split(gaes, df_train, df_eval, scaler):
    X_do = gaes.net.get_result(torch.Tensor(df_train.iloc[:, :-3].values.astype(np.float32).reshape(-1, 20, 1)).cuda(),
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
        torch.Tensor(df_eval.iloc[:, :-3].values.astype(np.float32).reshape(-1, 20, 1)).cuda(),
        do=1).detach().cpu().numpy().reshape(-1, 20)[:, 1:]
    eval_X_or = df_eval.iloc[:, 1:-3].values.astype(np.float32)
    lst_temp = [1 for _ in range(len(eval_X_do))]
    lst_temp.extend([0 for _ in range(len(eval_X_or))])
    eval_X = scaler.transform(np.concatenate((eval_X_do, eval_X_or), axis=0))

    eval_iter = DataLoader(CFDataset(torch.tensor(eval_X.astype(np.float32)), torch.LongTensor(lst_temp)),
                           batch_size=32, shuffle=False)

    return train_iter, eval_iter

def get_retrain_result(gaes, aae_trainer, df_test, df_test_cf=[], ratio=1, scaler=None):
    R_aae = aae_trainer.max_dist * ratio
    test_iter = DataLoader(scaler.transform(df_test.iloc[:, 1:-3].values.astype(np.float32)), batch_size=32, shuffle=False)
    lst_pred, lst_score = aae_trainer._evaluation(test_iter, df_test['label'], r=R_aae)
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

    if len(df_test_cf) >= 1:
        test_iter = DataLoader(scaler.transform(df_test_cf.iloc[:, 1:-3].values.astype(np.float32).reshape(-1, 19)), batch_size=32,
                               shuffle=False)
        lst_pred, lst_score = aae_trainer._evaluation(test_iter, df_test_cf['label'], r=R_aae)
        print('CF')
        print(classification_report(y_true=df_test_cf['label'], y_pred=lst_pred, digits=5))
        print(confusion_matrix(y_true=df_test_cf['label'], y_pred=lst_pred))
        print(f"AUC-PR: {average_precision_score(y_true=df_test_cf['label'], y_score=lst_score)}")
        print(f"AUC-ROC: {roc_auc_score(y_true=df_test_cf['label'], y_score=lst_score)}")
        df_ad['pred_cf'] = lst_pred

    return df_ad


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


def get_fairness_result_correct(df_org, df_ad, cf=0):
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
        df_org_c = df_org.loc[np.where(df_org['pred'].values == df_org['label'].values)]
        df_org_c = df_org_c.loc[df_org_c['label'] == 1]
        df_org_cf = df_org_c.groupby(['cf_changed']).count().reset_index(drop=False)
        before_cf = sum(df_org_cf.loc[df_org_cf['cf_changed'] != 0]['label'].values)
        df_ad_c = df_ad.loc[np.where(df_ad['pred'].values == df_ad['label'].values)]
        df_ad_c = df_ad_c.loc[df_ad_c['label'] == 1]
        df_ad_cf = df_ad_c.groupby(['cf_changed']).count().reset_index(drop=False)
        after_cf = sum(df_ad_cf.loc[df_ad_cf['cf_changed'] != 0]['label'].values)
        print('Results for CF samples')
        print(before_cf)
        print(after_cf)
        print(len(df_org_c))
        print(len(df_ad_c))
        print(f'Without fair, the prediction changed: {before_cf / len(df_org_c)}')
        print(f'With fair, the prediction changed: {after_cf / len(df_ad_c)}')
