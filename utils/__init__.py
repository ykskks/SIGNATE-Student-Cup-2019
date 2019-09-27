import logging

import pandas as pd
from lightgbm.callback import _format_eval_result 


def update_tracking(model_id, field, value, csv_file='logs/history.csv',
                    integer=False, digits=None):
    try:
        df = pd.read_csv(csv_file, index_col=[0])
    except:
        df = pd.DataFrame()

    if integer:
        value = round(value)
    elif digits is not None:
        value = round(value, digits)
    df.loc[model_id, field] = value # Model number is index
    df.to_csv(csv_file)


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback