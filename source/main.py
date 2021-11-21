import numpy as np
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import LabelEncoder
from data_loader import load_dataset
from model_builder import HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER
from hyperparameter_tuner import run


if __name__ == '__main__':
    session_num = 0

    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            for optimizer in HP_OPTIMIZER.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: dropout_rate,
                    HP_OPTIMIZER: optimizer,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hyperparameters_tuning/' + run_name, hparams)
                session_num += 1