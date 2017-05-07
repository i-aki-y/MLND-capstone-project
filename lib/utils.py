from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from . import cols

Dataset = namedtuple("Dataset", ["name", "df_all", "df_dev", "df_train", "df_valid", "df_devtest", "df_test"])


def get_df_all():
    """Load local csv files and returns dataframes

    This loads csv files located in local input directory.
    Data are returned as dataframe object.
    Test and training and label data are merged into single dataframes.

    Returns:
      (df_sj, df_iq): Tupple of dataframe each one is correspond to different city.


    """
    df_test = pd.read_csv("inputs/dengue_features_test.csv")
    df_train = pd.read_csv("inputs/dengue_features_train.csv")
    df_label = pd.read_csv("inputs/dengue_labels_train.csv")
    df_train = pd.merge(df_train, df_label, on=["city", "year", "weekofyear"])

    df_test["total_cases"] = None
    df_all = pd.concat([df_train, df_test] )
    df_all["total_cases"] = pd.to_numeric(df_all["total_cases"])
    df_all_sj = df_all[df_all.city == 'sj'].sort_values(by=["week_start_date"]).reset_index(drop=True)
    df_all_iq = df_all[df_all.city == 'iq'].sort_values(by=["week_start_date"]).reset_index(drop=True)

    return (df_all_sj, df_all_iq)


def fill_na_by_col(df, col_dest, col_source):
    """Fill NA value with other column's value

    This copies value of col_source into col_dest when the col_dest have NA value
    If the col_dest have any value (not NA) nothing is done.

    Args:

      df: Dataframe which have NA values.
      col_dest: A column name which have NA values to be filled.
      col_source: A column name whose value is set to the col_dest if the col_dest value have NA.
    """
    is_null = df[col_dest].isnull()
    df.loc[is_null, col_dest] = df.loc[is_null, col_source]
    return df


def fill_na_by_interpolate(df, cols):
    """ Fill NA value with values of nearest neighber's interpolation.

    Args:
      df: Dataframe which have NA values.
      cols: Column name of dataframe to be filled.
    """
    for col in cols:
        df[col] = df[col].interpolate()


def fill_na(df):
    """ Fill NA value

    This fills NA values by two different ways.
    First, fills NA of some selected columns by using other column value.
    Second, fills NA by interpolation.

    Args:
      df: Dataframe which have NA values.
    """
    fill_na_by_col(df, "ndvi_ne", "ndvi_nw")
    col_feats = cols.ndvi + cols.precipitation + cols.reanalysis + cols.station
    fill_na_by_interpolate(df, col_feats)


def set_month(df):
    """Add month columns

    Args:
      df: Dataframe
    """
    df["month"] = pd.to_datetime(df.week_start_date).dt.month


def calc_lag_column(df, cols, n):
    """Add new columns with time lagged values.
    New column names with time lag values are given by '{colname}_lag{n}'.

    Args:
      df: Dataframe.
      cols: Column name list to calculate lag.
      n: lags are calculated from 1 to n (including n).
    """
    for i in range(n):
        for col in cols:
            col_lag = lag_column_name_at(col, i+1)
            df[col_lag] = df[col].shift(i+1)

def calc_lag_column_by_list(df, cols, nlist):
    """Add new columns with time lagged values.
    New column names with time lag values are given by '{colname}_lag{n}'.

    Args:
      df: Dataframe.
      cols: Column name list to calculate lag.
      nlist([int]): List containing lag step numbers.
    """
    for i in nlist:
        for col in cols:
            col_lag = lag_column_name_at(col, i)
            df[col_lag] = df[col].shift(i)

def lag_column_name_at(col, i):
    """Generage new time lagged column's name.
    This returns column name with '{col}_lag{i}' fromat.

    Args:
      col: Original column name.
      i: time step of lag

    Returns:
      newcolname: String with the format '{col}_lag{i}' fromat.
    """
    return col + "_lag{0}".format(i)


def calc_log1p(df):
    """Generate new target column of `log(1 + x)`.

    Calculate `log(1 + x)` from target column and add it into dataframe.

    Args:
      df: Dataframe
    """
    df[cols.log1p_target] = np.log1p(df[cols.target])


def split_data(df_all_sj, df_all_iq, print_info=True):
    """Split dataframe into training, valid, devtest, test dataframes.

    This split dataframe into training, valid, devtest and test dataframes.
    Dataframe is split according to the value of the `week_start_date`.
    Returned object is dataset which contains split dataframes

    Args:
     df_all_sj: Dataframe of San Juan
     df_all_iq: Dataframe of Iquitos
     print_info: Whether show dataframe info or not.
    """

    sj_valid_s = "2003-04-22"
    sj_devtest_s = "2006-04-22" # 2008-04-22
    iq_valid_s = "2007-06-25"
    iq_devtest_s = "2009-06-25" # 2010-06-25

    is_test_sj = df_all_sj.total_cases.isnull()
    is_test_iq = df_all_iq.total_cases.isnull()

    is_train_sj = (df_all_sj.week_start_date < sj_valid_s) & ~is_test_sj
    is_valid_sj = (df_all_sj.week_start_date >= sj_valid_s) & (df_all_sj.week_start_date < sj_devtest_s) & ~is_test_sj
    is_devtest_sj = (df_all_sj.week_start_date >= sj_devtest_s) & ~is_test_sj

    is_train_iq = (df_all_iq.week_start_date < iq_valid_s) & ~is_test_iq
    is_valid_iq = (df_all_iq.week_start_date >= iq_valid_s) & (df_all_iq.week_start_date < iq_devtest_s) & ~is_test_iq
    is_devtest_iq = (df_all_iq.week_start_date >= iq_devtest_s) & ~is_test_iq

    df_test_sj = df_all_sj[is_test_sj]
    df_dev_sj = df_all_sj[~is_test_sj]
    df_train_sj = df_all_sj[is_train_sj]
    df_valid_sj = df_all_sj[is_valid_sj]
    df_devtest_sj = df_all_sj[is_devtest_sj]

    df_test_iq = df_all_iq[is_test_iq]
    df_dev_iq = df_all_iq[~is_test_iq]
    df_train_iq = df_dev_iq[is_train_iq]
    df_valid_iq = df_dev_iq[is_valid_iq]
    df_devtest_iq = df_dev_iq[is_devtest_iq]

    df_all_sj["data_type"] = 0
    df_all_sj.loc[is_valid_sj ,"data_type"] = 1
    df_all_sj.loc[is_devtest_sj ,"data_type"] = 2
    df_all_sj.loc[is_test_sj ,"data_type"] = 3

    df_all_iq["data_type"] = 0
    df_all_iq.loc[is_valid_iq ,"data_type"] = 1
    df_all_iq.loc[is_devtest_iq ,"data_type"] = 2
    df_all_iq.loc[is_test_iq ,"data_type"] = 3

    assert sum([len(df) for df in [df_train_sj, df_valid_sj, df_train_iq, df_valid_iq]]), len(df_train)

    if(print_info):
        print("sj train: {0:d} lines\t valid: {1:d} lines\t devtest: {2:d} lines\t test: {3:d} lines".format(len(df_train_sj),
                                                                                                             len(df_valid_sj),
                                                                                                             len(df_devtest_sj),
                                                                                                             len(df_test_sj)))
        print("iq train: {0:d} lines\t valid: {1:d} lines\t devtest: {2:d} lines\t test: {3:d} lines".format(len(df_train_iq),
                                                                                                             len(df_valid_iq),
                                                                                                             len(df_devtest_iq),
                                                                                                             len(df_test_iq)))
        print("sj valid: {0:.2f}%\niq valid: {1:.2f}%".format(len(df_valid_sj)/(len(df_train_sj) + len(df_valid_sj)), len(df_valid_iq)/(len(df_train_iq) + len(df_valid_iq))))

    ds_sj = Dataset(
        name = "sj",
        df_all = df_all_sj,
        df_dev=df_dev_sj,
        df_train=df_train_sj,
        df_valid=df_valid_sj,
        df_devtest=df_devtest_sj,
        df_test = df_test_sj
    )

    ds_iq = Dataset(
        name = "iq",
        df_all = df_all_iq,
        df_dev=df_dev_iq,
        df_train=df_train_iq,
        df_valid=df_valid_iq,
        df_devtest=df_devtest_iq,
        df_test = df_test_iq
    )
    return (ds_sj, ds_iq)


def show_dataset_info(ds):
    """Print breaf summary of dataframe contained dataset.

    Args:
      ds: Dataset
    """
    print("df_all       :\t{0}\t{1}".format(*get_date_ends(ds.df_all)))
    print("df_dev     :\t{0}\t{1}".format(*get_date_ends(ds.df_dev)))
    print("df_train    :\t{0}\t{1}".format(*get_date_ends(ds.df_train)))
    print("df_valid    :\t{0}\t{1}".format(*get_date_ends(ds.df_valid)))
    print("df_devtest:\t{0}\t{1}".format(*get_date_ends(ds.df_devtest)))
    print("df_test    :\t{0}\t{1}".format(*get_date_ends(ds.df_test)))


def get_date_ends(df):
    """Returns `week_start_date` ends of given dataframe.

    Args:
      df: Dataframe containing `week_start_date` column

    Returns:
      (start_date, end_date): Tupple of datetimes
    """

    return (df.week_start_date.iloc[0], df.week_start_date.iloc[-1])



def percentile(n):
    """Returns `n` percentile function
    Make function which calculate `n` percentile.

    Args:
      n: percent

    Returns:
      p: `n` persentile function
    """
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def set_monthly(df):
    """Add monthly statistics columns
    Calculate monthly statistics and add the calculated values as new columns

    Args:
      df: Dataframe

    Returns:
      df: Dataframe including monthly statistics
    """
    df["month"] = pd.to_datetime(df.week_start_date).dt.month
    df_m = df[~df.total_cases.isnull()].groupby("month")[cols.target].agg(
        [np.mean,
         np.median,
         np.std,
         percentile(75),
         percentile(95)
        ])
    df_m.columns = ["tc_month_mean", "tc_month_median", "tc_month_std", "tc_month_per75", "tc_month_per95"]
    return df.merge(df_m.reset_index(), on="month").sort_values(by=["week_start_date"]).reset_index(drop=True)


def get_autocorr_at(df, cols, lag=1, higher=0.6, lower=0.25):
    """Calculate autocorrelation at `lag` and retuns column name list seperated by levels

    Args:
      df: Dataframe
      cols: Column names to calculate autocorrelation
      lag: time step of autocorrelation lag
      higher: highter boundary to split results
      lower: lower boundary to split results

    Returns:
      (high_ac, middle_ac, low_ac): Tupple of column name list
    """
    ac_infos=[]
    for col in cols:
        info = {}
        info["col"] = col
        info["lag"] = lag
        info["val"] = df[col].autocorr(lag=1)
        ac_infos.append(info)

    high_ac = [item["col"] for item in ac_infos if item["val"] > higher ]
    middle_ac = [item["col"] for item in ac_infos if item["val"] <= higher and item["val"] > lower]
    low_ac = [item["col"] for item in ac_infos if item["val"] <= lower]

    return high_ac, middle_ac, low_ac


def get_timestamped_name(name):
    """Add timesstamp string to the given name

    Args:
      name: original name

    Retuns:
      timestamped_name: name attached timestamp
    """
    from datetime import datetime

    timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
    return "{0}_{1}".format(timestamp, name)

def save_result(create_predict_func, model_sj, model_iq, col_feats_sj, col_feats_iq, df_test_sj, df_test_iq, res_name):
    """Save results(models, submission, related information)

    Args:

    Returns:
    """
    sj_predictions = create_predict_func(model_sj, col_feats_sj, df_test_sj)
    iq_predictions = create_predict_func(model_iq, col_feats_iq, df_test_iq)

    submission = pd.read_csv("./inputs/submission_format.csv",
                             index_col=[0, 1, 2])

    uniq_fname = get_timestamped_name(res_name)

    sub_name = "{0}_{1}".format(uniq_fname, "sub.csv")

    submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
    submission.to_csv("./outputs/" + sub_name)

    sj_model_name = "{0}_{1}".format(uniq_fname, "mdl_sj.pkl")
    iq_model_name = "{0}_{1}".format(uniq_fname, "mdl_iq.pkl")

    joblib.dump(model_sj, "./outputs/" + sj_model_name)
    joblib.dump(model_iq, "./outputs/" + iq_model_name)

    sj_feats_name = "{0}_{1}".format(uniq_fname, "feats_sj.pkl")
    iq_feats_name = "{0}_{1}".format(uniq_fname, "feats_iq.pkl")

    joblib.dump(col_feats_sj, "./outputs/" + sj_feats_name)
    joblib.dump(col_feats_iq, "./outputs/" + iq_feats_name)

def load_model(uniq_fname):

    sj_model_name = "{0}_{1}".format(uniq_fname, "mdl_sj.pkl")
    iq_model_name = "{0}_{1}".format(uniq_fname, "mdl_iq.pkl")

    sj_feats_name = "{0}_{1}".format(uniq_fname, "feats_sj.pkl")
    iq_feats_name = "{0}_{1}".format(uniq_fname, "feats_iq.pkl")

    sj_model = joblib.load("./outputs/" + sj_model_name)
    iq_model = joblib.load("./outputs/" + iq_model_name)

    sj_feats = joblib.load("./outputs/" + sj_feats_name)
    iq_feats = joblib.load("./outputs/" + iq_feats_name)

    return {"sj_model": sj_model, "iq_model": iq_model, "sj_feats": sj_feats, "iq_feats": iq_feats}
