import pandas as pd

from utils import config_utils, utils, synthetic_dataset, adult_config, compas_config
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models import PCA


def main():
    parser = config_utils.get_args()
    args = parser.parse_args()
    options = vars(args)

    # Reproducibility
    utils.set_seed(options['seed'])

    # Get dataset
    print('Start loading synthetic data')
    df_train, df_test, df_test_cf = utils.load_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.iloc[:,1:-3].values)
    X_test = scaler.transform(df_test.iloc[:, 1:-3].values)
    y_test = df_test['label'].values
    X_test_cf = scaler.transform(df_test_cf.iloc[:, 1:-3].values)
    y_test_cf = df_test_cf['label'].values
    pca = PCA.PCA()
    pca.fit(X_train)
    y_pred, y_score = pca.predict(X_test)
    y_pred_cf, y_score_cf = pca.predict(X_test_cf)

    print('Observed Results:')
    print(classification_report(y_true=y_test, y_pred=y_pred, digits=5))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"AUC-PR: {average_precision_score(y_true=y_test, y_score=y_score)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=y_test, y_score=y_score)}")

    print('Counterfactual Results:')
    print(classification_report(y_true=y_test_cf, y_pred=y_pred_cf, digits=5))
    print(confusion_matrix(y_true=y_test_cf, y_pred=y_pred_cf))
    print(f"AUC-PR: {average_precision_score(y_true=y_test_cf, y_score=y_score_cf)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=y_test_cf, y_score=y_score_cf)}")

    df = pd.DataFrame()
    df['pred'] = y_pred
    df['pred_cf'] = y_pred_cf
    df['cf_changed'] = df['pred_cf'] - df['pred']
    total = len(df)
    df_cf = df.groupby(['cf_changed']).count().reset_index(drop=False)
    pca_cf = sum(df_cf.loc[df_cf['cf_changed'] != 0]['pred_cf'].values)
    print(f'PCA the prediction changed: {pca_cf / total}')

    # Adult dataset
    print('-' * 60)
    print('Results for Adult dataset')
    parser = adult_config.get_args()
    args = parser.parse_args()
    options = vars(args)

    # Reproducibility
    utils.set_seed(options['seed'])

    # Get dataset
    print('Start loading synthetic data')
    df_train, df_test, df_test_cf = utils.load_data(1)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_train.iloc[:, 1:-1].values)
    X_test = scaler.transform(df_test.iloc[:, 1:-1].values)
    y_test = df_test['y'].values
    X_test_cf = scaler.transform(df_test_cf.values)
    y_test_cf = df_test['y'].values
    pca = PCA.PCA(threshold=0.2)
    pca.fit(X_train)
    y_pred, y_score = pca.predict(X_test)
    y_pred_cf, y_score_cf = pca.predict(X_test_cf)

    print('Observed Results:')
    print(classification_report(y_true=y_test, y_pred=y_pred, digits=5))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"AUC-PR: {average_precision_score(y_true=y_test, y_score=y_score)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=y_test, y_score=y_score)}")

    print('Counterfactual Results:')
    print(classification_report(y_true=y_test_cf, y_pred=y_pred_cf, digits=5))
    print(confusion_matrix(y_true=y_test_cf, y_pred=y_pred_cf))
    print(f"AUC-PR: {average_precision_score(y_true=y_test_cf, y_score=y_score_cf)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=y_test_cf, y_score=y_score_cf)}")

    df = pd.DataFrame()
    df['pred'] = y_pred
    df['pred_cf'] = y_pred_cf
    df['cf_changed'] = df['pred_cf'] - df['pred']
    total = len(df)
    df_cf = df.groupby(['cf_changed']).count().reset_index(drop=False)
    pca_cf = sum(df_cf.loc[df_cf['cf_changed'] != 0]['pred_cf'].values)
    print(f'PCA the prediction changed: {pca_cf / total}')

    # Compas dataset
    print('-' * 60)
    print('Results for Compas dataset')
    parser = compas_config.get_args()
    args = parser.parse_args()
    options = vars(args)

    # Reproducibility
    utils.set_seed(options['seed'])

    # Get dataset
    print('Start loading synthetic data')
    df_train, df_test, df_test_cf = utils.load_data(2)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_train.iloc[:, 1:-1].values)
    X_test = scaler.transform(df_test.iloc[:, 1:-1].values)
    y_test = df_test['y'].values
    X_test_cf = scaler.transform(df_test_cf.values)
    y_test_cf = df_test['y'].values
    pca = PCA.PCA(threshold=0.2)
    pca.fit(X_train)
    y_pred, y_score = pca.predict(X_test)
    y_pred_cf, y_score_cf = pca.predict(X_test_cf)

    print('Observed Results:')
    print(classification_report(y_true=y_test, y_pred=y_pred, digits=5))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"AUC-PR: {average_precision_score(y_true=y_test, y_score=y_score)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=y_test, y_score=y_score)}")

    print('Counterfactual Results:')
    print(classification_report(y_true=y_test_cf, y_pred=y_pred_cf, digits=5))
    print(confusion_matrix(y_true=y_test_cf, y_pred=y_pred_cf))
    print(f"AUC-PR: {average_precision_score(y_true=y_test_cf, y_score=y_score_cf)}")
    print(f"AUC-ROC: {roc_auc_score(y_true=y_test_cf, y_score=y_score_cf)}")

    df = pd.DataFrame()
    df['pred'] = y_pred
    df['pred_cf'] = y_pred_cf
    df['cf_changed'] = df['pred_cf'] - df['pred']
    total = len(df)
    df_cf = df.groupby(['cf_changed']).count().reset_index(drop=False)
    pca_cf = sum(df_cf.loc[df_cf['cf_changed'] != 0]['pred_cf'].values)
    print(f'PCA the prediction changed: {pca_cf / total}')

if __name__ == '__main__':
    main()