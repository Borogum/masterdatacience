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
        machine_id = str(uuid.uuid4())
        telemetry_notifier = CsvNotifier(config['telemetry_path'], 'telemetry_%s' % machine_id, buffer_size=None)
        event_notifier = CsvNotifier(config['event_path'], 'event_%s' % machine_id, buffer_size=None)
        facility.add(
            Machine(name=machine_id, telemetry_notifier=telemetry_notifier, event_notifier=event_notifier,
                    **config))
    return facility


def flush_all(facilities):

    for facility in facilities:
        for machine in facility.machines:
            machine.telemetry_notifier.flush()
            machine.event_notifier.flush()


if __name__ == '__main__':

    start_time = time.time()
    data_path = 'data'
    duration = 2 * 30 * 24 * 60 * 60  # two months

    cl = Clock()

    facilities_configs = [
        {'temperature': 15,
         'pressure': 98,
         'cycle_length_min': 1,
         'cycle_length_max': 5,
         'cycle_duration': 60,
         'machines_per_batch': 5,
         'operational_speed_min': 800,
         'operational_speed_max': 900,
         'clock': cl,
         'batch_time': 3600,
         'machine_count': 10,  # No facility attr
         'ttf1_min': 5000,
         'ttf1_max': 50000,
         'ttf2_min': 500,
         'ttf2_max': 90000,
         'temperature_max': 100,
         'pressure_factor': 1.8,
         'telemetry_path': os.path.join(data_path, 'facility_a', 'telemetry'),
         'event_path': os.path.join(data_path, 'facility_a', 'event'),
         },
    ]

    facilities_list = [create_facility(cfg) for cfg in facilities_configs]

    completion_str = 'Progress: %.2f%%' % 0
    print(completion_str, end='')

    for i in range(duration):

        print('\b' * len(completion_str), end='')
        completion_str = 'Progress: %.2f%%' % (100. * i / duration)
        print(completion_str, end='')

        for n, f in enumerate(facilities_list):
            next(f)
        next(cl)

    flush_all(facilities_list)

    print('\n--- %.2f seconds ---' % (time.time() - start_time))
