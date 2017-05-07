
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA

from . import utils
from . import cols

def preprocess():

    # load dateset
    df_all_sj, df_all_iq = utils.get_df_all()
    df_all_list = [df_all_sj, df_all_iq]

    for df_all in df_all_list:
        # Fill NA
        utils.fill_na(df_all)
        # set month
        utils.set_month(df_all)

    # rolling average
    for df_all in df_all_list:
        for col in cols.orig_feats:
            df_all["roll4_" + col] = df_all[col].rolling(4, min_periods=1).mean()
    col_roll_feats = cols.get_roll_feats(cols.orig_feats, 4)

    lags = [4, 8, 12, 16, 20]
    # add time lag of rolling average
    for df_all in df_all_list:
        utils.calc_lag_column_by_list(df_all, col_roll_feats, lags)
        df_all.bfill(inplace=True)

    col_roll_lag_feats = cols.get_lag_feats(col_roll_feats, lags)

    # splitdata(to take df_train)
    ds_sj, ds_iq = utils.split_data(*df_all_list, print_info=False)

    col_all_feats = cols.orig_feats + col_roll_feats + col_roll_lag_feats
    # scale features
    scaler_sj = preprocessing.StandardScaler().fit(ds_sj.df_train[col_all_feats])
    scaler_iq = preprocessing.StandardScaler().fit(ds_iq.df_train[col_all_feats])

    col_scaled_all_feats = cols.get_scaled_feats(col_all_feats)
    for df_all, scaler in zip(df_all_list, [scaler_sj, scaler_iq]):
        df_scaled = pd.DataFrame(scaler.transform(df_all[col_all_feats]), columns = col_scaled_all_feats)
        df_all[col_scaled_all_feats] = df_scaled[col_scaled_all_feats]

    # re-create splitdata (to apply scaled features)
    ds_sj, ds_iq = utils.split_data(*df_all_list, print_info=False)

    # pca features
    pca_sj = PCA(whiten=False, n_components=7)
    pca_iq = PCA(whiten=False, n_components=7)

    col_scaled_orig_feats = cols.get_scaled_feats(cols.orig_feats)
    pca_sj.fit(ds_sj.df_train[col_scaled_orig_feats].values)
    pca_iq.fit(ds_iq.df_train[col_scaled_orig_feats].values)

    col_pca_feats = cols.get_pca_feats(pca_sj.n_components_)
    for df_all, pca in zip(df_all_list, [pca_sj, pca_iq]):
        df_pca = pd.DataFrame(pca.transform(df_all[col_scaled_orig_feats]), columns = col_pca_feats)
        df_all[col_pca_feats] = df_pca[col_pca_feats]

    # Add time lagged columns for RNN model
    ts = 64
    for df in df_all_list:
        for i in range(ts):
            # for col in col_scaled_all_feats:
            for col in col_scaled_orig_feats:
                col_lag = col + "_lag{0}".format(i+1)
                df[col_lag] = df[col].shift(i+1)

    # re-create splitdata (to apply pca features)
    ds_sj, ds_iq = utils.split_data(*df_all_list)
    return ds_sj, ds_iq
