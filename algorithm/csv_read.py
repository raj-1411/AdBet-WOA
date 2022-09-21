'''

Created on Sep,2022

'''

import pandas as pd
import numpy as np


def file_read(args, root):

    file_name = ['train', 'validation', 'test']
    features_tuple = ()

    for p_fix in file_name:
        features_tuple = features_tuple + pd.read_csv(root+'{}_features.csv'.format(p_fix)).to_numpy()

    return features_tuple