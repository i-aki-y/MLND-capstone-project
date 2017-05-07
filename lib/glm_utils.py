from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from . import cols
from . import utils

def create_prediction(model, df, cols):
    """Returns prediction from given dataframe

    Args:
      model: Trained model
      df: Dataframe containing necessary features
      cols: Names of features required by model

    Returns:
      pred: Prediction result
    """
    return model.predict(df[cols]).astype(int)

def save_result(model_sj, model_iq, col_feats_sj, col_feats_iq, df_test_sj, df_test_iq, res_name):
    """Save results(models, submission, related information)

    Args:

    Returns:
    """
    sj_predictions = create_prediction(model_sj, df_test_sj, col_feats_sj)
    iq_predictions = create_prediction(model_iq, df_test_iq, col_feats_iq)

    submission = pd.read_csv("./inputs/submission_format.csv",
                             index_col=[0, 1, 2])

    uniq_fname = utils.get_timestamped_name(res_name)

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
    return utils.load_model(uniq_fname)
