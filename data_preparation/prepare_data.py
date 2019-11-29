import os
import glob
from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare(path, id, test_size=0.2, lookback=5):
    # Load telemetry data
    processed_filename = os.path.join(path, 'processed_%s.csv' % machine_id)
    processed_df = pd.read_csv(processed_filename, sep=',', decimal='.', dtype={'immediate_failure': np.int})
    processed_df.drop(['id', 'cycle_id', 'sequence_id', 'timestamp_start', 'timestamp_end', 'rul'], inplace=True,
                      axis=1)
    train, test = train_test_split(processed_df, shuffle=False, test_size=test_size)
    train = train[:-lookback + 1]  # remove last item for avoid overlapping
    class_counter = Counter(train['immediate_failure'])
    print(class_counter)
    # ss = StandardScaler()
    # X_train = ss.fit_transform(X_train)
    return train, test


if __name__ == '__main__':

    input_base_path = os.path.join('..', 'data_processing', 'data', 'facility_a')
    output_base_path = os.path.join('data', 'facility_a')
    for f in glob.glob(os.path.join(input_base_path, 'processed_*.csv')):
        machine_id = os.path.basename(f).split('.')[0].split('_')[1]
        print('Processing machine: %s ...' % machine_id)
        train_df, test_df = prepare(input_base_path, machine_id)
        train_df.to_csv(os.path.join(output_base_path, 'train', '%s.csv' % machine_id), index=False, sep=',', decimal='.')
        test_df.to_csv(os.path.join(output_base_path, 'test', '%s.csv' % machine_id), index=False, sep=',', decimal='.')
