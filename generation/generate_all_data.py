import os
import time
import glob
import argparse
import subprocess
from queue import Queue, Empty
from threading import Thread


def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate company data')
    parser.add_argument('path', type=str, help='Folder where configuration files live')
    args = parser.parse_args()

    q = Queue()
    processes = []
    for f in glob.glob(os.path.join(args.path, '*.cfg')):
        p = subprocess.Popen(['python', 'generate_data.py', f], stdout=subprocess.PIPE)
        processes.append(p)
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True  # thread dies with the program
        t.start()

    # Print info
    while True:
        finished_counter = 0
        for p in processes:
            if p.poll() is not None:  # Not finished
                finished_counter += 1
        try:
            while True:
                print(q.get_nowait().decode(), end='')
        except Empty:
            pass

        if finished_counter == len(processes):
            break
        time.sleep(1)
