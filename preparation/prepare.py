import os
import glob
import shutil
import argparse
import configparser
from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def join_split(folder, test_size=0.2, lookback=5):

    train_list = []
    test_list = []

    for f in glob.glob(os.path.join(folder, 'processed_*.csv')):
        df = pd.read_csv(f, sep=',', decimal='.', dtype={'immediate_failure': np.int})
        df.drop(['id', 'cycle_id', 'sequence_id', 'timestamp_start', 'timestamp_end', 'rul'], inplace=True, axis=1)
        train, test = train_test_split(df, shuffle=False, test_size=test_size)
        train = train[:-lookback + 1]
        train_list.append(train)
        test_list.append(test)

    return pd.concat(train_list, ignore_index=True), pd.concat(test_list, ignore_index=True)


def normalize(train, test):

    ss = StandardScaler()
    x_train_data = train[train.columns[:-1]]
    y_train_data = train[train.columns[-1]]
    x_test_data = test[test.columns[:-1]]
    y_test_data = test[test.columns[-1]]

    return ss.fit_transform(x_train_data), y_train_data.values, ss.transform(x_test_data), y_test_data.values


def smote(x, y, p=.2, seed=None):

    all_classes = Counter(y)
    minority_classes = all_classes.most_common()[1:]
    desired_minority_classes_size = int(y.shape[0] * p)
    sampling_strategy = dict((c[0], max(desired_minority_classes_size, c[1])) for c in minority_classes)
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=seed)
    x_res, y_res = sm.fit_sample(x, y)
    return x_res, y_res


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('config', type=str, help='Configuration file')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    config_test_size = config.getfloat('CONFIGURATION', 'test_size')
    config_lookback = config.getint('CONFIGURATION', 'lookback')
    config_p = config.getfloat('CONFIGURATION', 'p')
    path_in = config.get('CONFIGURATION', 'path_in')
    path_out = config.get('CONFIGURATION', 'path_out')

    directories = []
    for directory in glob.glob(os.path.join(path_in, '*', ''), recursive=False):
        directories.append(os.path.basename(os.path.normpath(directory)))

    for d in directories:

        input_base_path = os.path.join(path_in, d)
        output_base_path = os.path.join(path_out, d)

        # Create dirs
        if os.path.isdir(output_base_path):
            shutil.rmtree(output_base_path)
        os.makedirs(output_base_path)

        print('With %s:' % d)
        print('Joining all files and doing train-test split ... ', end='')
        train_df, test_df = join_split(input_base_path, test_size=config_test_size, lookback=config_lookback)
        print('Done!')
        print('Normalizing data ... ', end='')
        x_train, y_train, x_test, y_test = normalize(train_df, test_df)
        print('Done!')
        print('SMOTING ...', end='')
        x_train, y_train = smote(x_train, y_train, p=config_p)
        print('Done!')
        train_file = os.path.join(output_base_path, 'train.csv')
        test_file = os.path.join(output_base_path, 'test.csv')
        print('Saving files ...', end='')
        pd.DataFrame(data=np.append(x_train, y_train.reshape(-1, 1), axis=1)).to_csv(train_file, index=False,
                                                                                     decimal='.', sep=',')
        pd.DataFrame(data=np.append(x_test, y_test.reshape(-1, 1), axis=1)).to_csv(test_file, index=False,
                                                                                   decimal='.', sep=',')
        print('Done!')
