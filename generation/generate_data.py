import os
import uuid
import time
import shutil
import argparse
import configparser
from generation.simulation import Clock, Facility, Machine
from generation.notify import CsvNotifier


def create_facility(configuration):
    fac = Facility(**configuration)
    for _ in range(int(configuration['machine_count'])):
        machine_id = str(uuid.uuid4())
        telemetry_notifier = CsvNotifier(configuration['telemetry_path'], 'telemetry_%s' % machine_id, buffer_size=None)
        event_notifier = CsvNotifier(configuration['event_path'], 'event_%s' % machine_id, buffer_size=None)
        fac.add(
            Machine(name=machine_id, telemetry_notifier=telemetry_notifier, event_notifier=event_notifier,
                    **configuration))
    return fac


if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument('config', type=str, help='Facility configuration file')
    arg = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(arg.config)
    cl = Clock()
    facility_name = config['CONFIGURATION']['name']
    duration = config.getint('CONFIGURATION', 'simulation_time')
    config_dict = {'clock': cl}

    for key in config['CONFIGURATION']:
        if key not in ['name', 'simulation_time']:
            try:
                config_dict[key] = config.getfloat('CONFIGURATION', key)
            except ValueError as e:
                config_dict[key] = config.get('CONFIGURATION', key)

    facility = create_facility(config_dict)

    paths = (config_dict['telemetry_path'], config_dict['event_path'])

    # Create dirs structure

    for path in paths:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)

    progress_step = int(duration / 10)

    for i in range(duration):
        if i % progress_step == 0:
            print('Generating data for "%s" ... %.2f%%' % (facility_name, 100. * i / duration))
        next(facility)
        next(cl)
    print('Generating data for "%s" ... %.2f%%' % (facility_name, 100))

    print('Saving files for "%s" ... ' % facility_name)

    for machine in facility.machines:
        machine.telemetry_notifier.flush()
        machine.event_notifier.flush()

    print('All done for "%s". Task finished in %.2f seconds' % (facility_name, time.time() - start_time))
