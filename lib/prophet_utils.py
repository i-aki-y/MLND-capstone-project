import numpy as np
import pandas as pd

from . import utils

def create_prediction(fcst, df):
    y_pred = np.expm1(fcst[fcst.ds.isin(df.week_start_date)]._yhat.values).round().astype(int)
    return y_pred

def save_result(model_sj, model_iq, df_test_sj, df_test_iq, res_name):
    sj_predictions = create_prediction(model_sj, df_test_sj)
    iq_predictions = create_prediction(model_iq, df_test_iq)

    submission = pd.read_csv("./inputs/submission_format.csv",
                             index_col=[0, 1, 2])

    uniq_fname = utils.get_timestamped_name(res_name)

    sub_name = "{0}_{1}".format(uniq_fname, "sub.csv")

    submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
    submission.to_csv("./outputs/" + sub_name)
