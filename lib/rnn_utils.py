from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from sklearn.externals import joblib

from . import cols
from . import utils


def lag_col(i, cols=cols.orig_feats):
    """Get time lag column names
    """
    return ["{0}_lag{1}".format(col, i) for col in cols]

def make_rnn_x(df, ts=5, cols=cols.orig_feats):
    """Make RNN's input data
    This makes input data for RNN model.
    This assumes that the given dataframe have columns the name matches `{col}_log{i}`

    Returns:
      X: numpy array with shape (D, T, N)
         where D:sample num, T: time sequence, N: feature num.
    """
    ts_list = [df[lag_col(i, cols)].values for i in range(ts, 0, -1)]
    return np.transpose(ts_list, (1, 0, 2))


def get_X_y(df, ts_offset, ts_len, cols=cols.orig_feats):
    """Make X and y values for RNN
    """
    X = make_rnn_x(df[ts_offset:], ts_len, cols)
    y = df.total_cases.values[ts_offset:].reshape(-1, 1)
    return X, y



def create_prediction(model, df, cols):
    """Create prediction data
    """
    ts_len = model.input.shape[1].value
    X = make_rnn_x(df, ts_len, cols)
    return np.around(model.predict(X).squeeze()).astype(int)


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

    sj_model_name = "{0}_{1}".format(uniq_fname, "mdl_sj.hd5")
    iq_model_name = "{0}_{1}".format(uniq_fname, "mdl_iq.hd5")
    model_sj.save("./outputs/" + sj_model_name)
    model_iq.save("./outputs/" + iq_model_name)


    sj_feats_name = "{0}_{1}".format(uniq_fname, "feats_sj.pkl")
    iq_feats_name = "{0}_{1}".format(uniq_fname, "feats_iq.pkl")

    joblib.dump(col_feats_sj, "./outputs/" + sj_feats_name)
    joblib.dump(col_feats_iq, "./outputs/" + iq_feats_name)


def load_model(uniq_fname):
    from keras.models import load_model

    sj_model_name = "{0}_{1}".format(uniq_fname, "mdl_sj.hd5")
    iq_model_name = "{0}_{1}".format(uniq_fname, "mdl_iq.hd5")

    sj_feats_name = "{0}_{1}".format(uniq_fname, "feats_sj.pkl")
    iq_feats_name = "{0}_{1}".format(uniq_fname, "feats_iq.pkl")

    sj_model = load_model("./outputs/" + sj_model_name)
    iq_model = load_model("./outputs/" + iq_model_name)

    sj_feats = joblib.load("./outputs/" + sj_feats_name)
    iq_feats = joblib.load("./outputs/" + iq_feats_name)

    return {"sj_model": sj_model, "iq_model": iq_model, "sj_feats": sj_feats, "iq_feats": iq_feats}
