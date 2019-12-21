
import sys
import time
import argparse
import subprocess
import configparser


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Turn on servers')
    parser.add_argument('config', type=str, help='Configuration file')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    processes = []
    for section in config.sections():

        config_id = config.get(section, 'id')
        config_host = config.get(section, 'host')
        config_port = config.get(section, 'port')
        config_verbose = config.getboolean(section, 'verbose')
        config_train = config.get(section, 'train')
        config_test = config.get(section, 'test')

        # Start server
        command = ['python',
                   '../single_model/start_worker.py',
                   config_id,
                   config_host,
                   config_port,
                   config_train,
                   config_test,
                   ]

        if config_verbose:
            command.append('--verbose')

        print('Starting server %s ... ' % config_id, end='')
        processes.append(subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE))  # Don´t show default msg
        time.sleep(2)
        print('Done! (%s)' % processes[-1].pid)



    print ('Press Ctrl+C to stop ... ')
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("You pressed Ctrl+C!")
        for p in processes:
            p.terminate()
        sys.exit(0)


