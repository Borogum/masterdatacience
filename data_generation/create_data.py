"""
Units:
 * Temperature: Celsius degrees
 * Pressure: kPa
 * Time: seconds
"""

import os
import uuid
import time
from data_generation.simulation import Clock, Facility, Machine
from data_generation.notify import ConsoleNotifier, CsvNotifier, ParquetNotifier


def create_facility(config):
    facility = Facility(**config)
    for _ in range(config['machine_count']):
        facility.add(Machine(name=str(uuid.uuid4()), **config))
    return facility


if __name__ == '__main__':

    start_time = time.time()
    data_path = r'D:\Master Data Science\TFM\PEC\PEC3\data'
    duration = 1 * 30 * 24 * 60 * 60  # one month
    duration = 2 * 24 * 60 * 60  # one day
    cl = Clock()

    facilities_configs = [
        {'temperature': 15,
         'pressure': 98,
         'cycle_length_min': 1,
         'cycle_length_max': 5,
         'cycle_duration': 60,
         'machines_per_batch': 10,
         'operational_speed_min': 800,
         'operational_speed_max': 900,
         'clock': cl,
         'batch_time': 3600,
         'machine_count': 1000,  # No facility attr
         'ttf_min': 500,
         'ttf_max': 10000,
         'temperature_max': 100,
         'pressure_factor': 1.8,
         'telemetry_notifier': CsvNotifier(os.path.join(data_path, 'facility_a'), 'telemetry', buffer_size=None),
         'event_notifier': CsvNotifier(os.path.join(data_path, 'facility_a'), 'event', buffer_size=None),
         },
    ]

    facilities = [create_facility(cfg) for cfg in facilities_configs]
    notifiers = [cfg['telemetry_notifier'] for cfg in facilities_configs] + [cfg['event_notifier'] for cfg in
                                                                             facilities_configs]
    completion_str = 'Progress: %.2f%%' % 0
    print(completion_str, end='')

    for i in range(duration):

        print('\b' * len(completion_str), end='')
        completion_str = 'Progress: %.2f%%' % (100. * i / duration)
        print(completion_str, end='')

        for n, f in enumerate(facilities):
            if (i % (facilities_configs[n]['batch_time'] - 1)) == 0:
                for no in notifiers:
                    no.flush()
            next(f)

        next(cl)

    for no in notifiers:
        no.flush()

    print('\n--- %.2f seconds ---' % (time.time() - start_time))
