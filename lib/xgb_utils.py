from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error

from . import cols
from . import utils

def col_fscore(model, fmap):
    fscores = model.booster().get_fscore().items()
    return [(fmap[key], val) for key, val in fscores]

def plot_fscore(model, fmap, output=None, figsize=(12, 8)):

    fscores = list(model.booster().get_fscore().items())[:10]
    df_fscore = pd.DataFrame([(fmap[key], val) for key, val in fscores],
                             columns=["feature", "fscore"]).sort_values(by=["fscore"], ascending=False)
    df_fscore.plot.barh(x = "feature", y = "fscore", figsize=figsize)
    if output:
        plt.savefig(output, bbox_inches='tight')

def plot_hist(model, metric="mae"):
    hist = pd.DataFrame({
        "trn": model.evals_result_["validation_0"][metric],
        "vld": model.evals_result_["validation_1"][metric],
    }, dtype=np.float32)
    hist.plot()

def plot_predict(model, df_dev, col_target, col_feats):
    figs, axes = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
    # plot sj
    df_dev['fitted'] = model.predict(df_dev[col_feats].values)
    df_dev.fitted.plot(ax=axes, label="Predictions")
    df_dev[col_target].plot(ax=axes, label="Actual")

    plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
    plt.legend()


def get_scores(model, ds, col_target, col_feats):
    y_valid_pred = model.predict(ds.df_valid[col_feats].values)
    y_devtest_pred = model.predict(ds.df_devtest[col_feats].values)

    y_valid_true = ds.df_valid[col_target].values
    y_devtest_true = ds.df_devtest[col_target].values

    if col_target == cols.log1p_target:
        devtest_score = mean_absolute_error(np.expm1(y_devtest_true), np.expm1(y_devtest_pred))
        valid_score = mean_absolute_error(np.expm1(y_valid_true), np.expm1(y_valid_pred))
    else:
        devtest_score = mean_absolute_error(y_devtest_true, y_devtest_pred)
        valid_score = mean_absolute_error(y_valid_true, y_valid_pred)

    print("valid_score: {0}\tdevtest_score: {1}".format(valid_score, devtest_score))
    return (valid_score, devtest_score)

def create_prediction(model, df, cols):
    """Returns prediction from given dataframe

    Args:
      model: Trained model
      df: Dataframe containing necessary features
      cols: Names of features required by model

    Returns:
      pred: Prediction result
    """
    return model.predict(df[cols].values).astype(int)


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
