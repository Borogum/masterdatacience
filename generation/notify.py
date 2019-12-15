import os
import csv
from abc import ABC, abstractmethod
import pandas as pd


class Notifier(ABC):

    @abstractmethod
    def notify(self, msg):
        pass

    @abstractmethod
    def flush(self):
        pass


class ConsoleNotifier(Notifier):

    def notify(self, msg):
        print(msg)

    def flush(self):
        pass


class ParquetNotifier(Notifier):

    def __init__(self, path, name, buffer_size=1000):
        self.filename = os.path.join(path, name) + '_{}.parquet'
        self.buffer_size = buffer_size
        self.buffer = []
        self.counter = 0

    def __write(self):
        pd.DataFrame(self.buffer).to_parquet(self.filename.format(self.counter), engine='fastparquet')

    def notify(self, msg):
        if (self.buffer_size is not None) and (len(self.buffer) == self.buffer_size):
            self.__write()
            self.buffer = []
            self.counter += 1
        else:
            self.buffer.append(msg)

    def flush(self):
        if len(self.buffer) > 0:
            self.__write()
            self.buffer = []
            self.counter += 1


class CsvNotifier(Notifier):

    def __init__(self, path, name, buffer_size=1000, lineterminator='\n', delimiter=','):
        self.filename = os.path.join(path, name) + '_{}.csv'
        self.buffer_size = buffer_size
        self.lineterminator = lineterminator
        self.delimiter = delimiter
        self.buffer = []
        self.counter = 0

    def __write(self):
        with open(self.filename.format(self.counter), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.buffer[0].keys(), lineterminator=self.lineterminator,
                                    delimiter=self.delimiter)
            writer.writeheader()
            writer.writerows(self.buffer)

    def notify(self, msg):
        if (self.buffer_size is not None) and (len(self.buffer) == self.buffer_size):
            self.__write()
            self.buffer = []
            self.counter += 1
        else:
            self.buffer.append(msg)

    def flush(self):
        if len(self.buffer) > 0:
            self.__write()
            self.buffer = []
            self.counter += 1
