"""
Units:
 * Temperature: Celsius degrees
 * Pressure: kPa
 * Time: seconds

- Crear la medias, maximos, mínimos etc...
- Asignar id  de secuencia (global) y id de posicion (local)
- Poner RUL

"""

import os
import glob
import numpy as np
import pandas as pd


def process(path, machine_id, gap=30, w=7, rolling_size=5):
    # Load telemetry data
    telemetry_filename = os.path.join(path, 'telemetry', 'telemetry_%s_0.csv' % machine_id)
    telemetry_df = pd.read_csv(telemetry_filename, sep=',', decimal='.')

    # Create cycle grouping
    time = telemetry_df['time']
    time_diff = time - time.shift(periods=1, fill_value=time[0])
    telemetry_df['cycle_id'] = np.cumsum(time_diff > gap)

    # Aggregate by cycle
    telemetry_grouped_df = telemetry_df.groupby(by='cycle_id').agg(
        {'id': ['max'], 'time': ['min', 'max'], 'speed': ['mean'],
         'target_speed': ['max'], 'temperature': ['mean', 'max'], 'pressure': ['mean', 'max']})
    telemetry_grouped_df.columns = ['id', 'timestamp_start', 'timestamp_end', 'speed_avg', 'target_speed_max',
                                    'temperature_avg', 'temperature_max', 'pressure_avg', 'pressure_max']
    telemetry_grouped_df.reset_index(inplace=True)

    # Load event data if exists (if not, create a empty dataframe)
    event_filename = os.path.join(path, 'event', 'event_%s_0.csv' % machine_id)
    if os.path.isfile(event_filename):
        event_df = pd.read_csv(event_filename, sep=',', decimal='.')
    else:
        event_df = pd.DataFrame(columns=['id', 'time', 'code', 'severity'])

    # Filter and select event data
    critical_df = event_df[event_df['severity'] == 'CRITICAL'][['time', 'code']]
    # Join data
    critical_df['time'] -= 1
    telemetry_event_df = telemetry_grouped_df.merge(critical_df, how='left', left_on='timestamp_end', right_on='time')
    telemetry_event_df['sequence_id'] = np.cumsum(
        pd.notnull(telemetry_event_df['time'].shift(periods=1, fill_value=None)))

    telemetry_event_df = telemetry_event_df[list(telemetry_grouped_df.columns) + ['sequence_id', 'code']]
    grouped_by_sequence_id = telemetry_event_df.groupby('sequence_id')
    # Calculate RUL (Remaining Useful Life)
    telemetry_event_df['rul'] = grouped_by_sequence_id['cycle_id'].transform(lambda x: len(x) - x + x.min() - 1)

    # Failure label for all records

    telemetry_event_df['upcoming_failure'] = grouped_by_sequence_id['code'].transform(
        lambda x: [x.iloc[-1] if not pd.isnull(x.iloc[-1]) else ''] * len(x))

    # Future horizon failure
    telemetry_event_df['immediate_failure'] = np.where(telemetry_event_df['rul'] < w,
                                                       telemetry_event_df['upcoming_failure'], '')

    # Calculate rolling features
    for col in ['temperature_avg', 'temperature_max', 'pressure_avg', 'pressure_max']:
        telemetry_event_df[col + '_avg'] = grouped_by_sequence_id[col].transform(
            lambda x: x.rolling(rolling_size).mean())

    # Filter non valid rolling features
    telemetry_event_df.dropna(subset=['temperature_max_avg'], inplace=True)

    # Select columns
    telemetry_event_df = telemetry_event_df[['id', 'cycle_id', 'sequence_id', 'timestamp_start', 'timestamp_end',
                                             'target_speed_max', 'speed_avg', 'temperature_avg', 'temperature_max',
                                             'temperature_avg_avg', 'temperature_max_avg', 'pressure_avg',
                                             'pressure_max', 'pressure_avg_avg', 'pressure_max_avg', 'rul',
                                             'immediate_failure']]

    return telemetry_event_df


if __name__== '__main__':

    input_base_path = os.path.join('..', 'data_generation', 'data', 'facility_a')
    output_base_path = os.path.join('data', 'facility_a')
    event_dir = os.path.join(input_base_path, 'event')
    telemetry_dir = os.path.join(input_base_path, 'telemetry')

    for f in glob.glob(os.path.join(telemetry_dir, 'telemetry_*.csv')):
        machine_id = os.path.basename(f).split('_')[1]
        print('Processing machine: %s ...' % machine_id)
        df = process(input_base_path, machine_id)
        df.to_csv(os.path.join(output_base_path, 'processed_%s.csv' % machine_id), index=False, sep=';', decimal='.')