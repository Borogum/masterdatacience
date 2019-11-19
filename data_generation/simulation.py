import numpy as np


class BrokenMachine(Exception):
    pass


class Clock(object):
    """Seconds"""

    __t = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.__t += 1
        return self.__t

    def now(self):
        return self.__t


class HealIndex(object):

    def __init__(self, name, ttf, d=0.05, a=-0.3, b=0.2, th=0):
        self.__name = name
        self.__t = ttf  # time to fail
        self.__d = d
        self.__a = a
        self.__b = b
        self.__th = np.max(th, 0)  # threshold

    def __iter__(self):
        return self

    def __next__(self):
        t = self.__t
        h = 1 - self.__d - np.exp(self.__a * self.__t ** self.__b)
        if h < self.__th:
            raise BrokenMachine(self.__name)
        self.__t -= 1
        return t, h


class Machine(object):

    def __init__(self, name, ttf1_min=500, ttf1_max=50000, ttf2_min=500, ttf2_max=50000, temperature_max=120,
                 ambient_temperature=20, ambient_pressure=101, pressure_factor=2, telemetry_notifier=None,
                 event_notifier=None, clock=None,
                 speed_threshold=20, **kwargs):
        self.name = name
        self.ttf1_min = ttf1_min
        self.ttf1_max = ttf1_max
        self.ttf2_min = ttf2_min
        self.ttf2_max = ttf2_max
        self.telemetry_notifier = telemetry_notifier
        self.event_notifier = event_notifier
        self.clock = clock
        self.h1 = HealIndex('F1', np.random.randint(self.ttf1_min, self.ttf1_max))
        self.h2 = HealIndex('F2', np.random.randint(self.ttf2_min, self.ttf2_max))
        self.ambient_temperature = ambient_temperature
        self.ambient_pressure = ambient_pressure
        self.speed_threshold = speed_threshold
        self.temperature_max = temperature_max
        self.pressure_factor = pressure_factor
        self.target_speed = 0
        self.maintenance = 0
        self.speed = 0
        self.temperature = 0
        self.pressure = 0

    def __repr__(self):
        return self.name

    def __iter__(self):
        return self

    @staticmethod
    def __g(v, min_v, max_v, target, rate):
        delta = (target - v) * rate
        return max(min(v + delta, max_v), min_v)

    @staticmethod
    def __noise(magnitude):
        return np.random.uniform(-magnitude, magnitude)

    def __now(self):
        return self.clock.now() if self.clock else None

    def get_status(self):
        return {
            'id': str(self),
            'time': self.__now(),
            'speed': self.speed + self.__noise(5),
            'target_speed': self.target_speed,
            'temperature': self.temperature + self.__noise(0.1),
            'pressure': self.pressure + self.__noise(20),
            'ambient_temperature': self.ambient_temperature + self.__noise(0.1),
            'ambient_pressure': self.ambient_pressure + self.__noise(0.1),
        }

    def repair(self, seconds):
        self.maintenance = seconds
        self.h1 = HealIndex('F1', np.random.randint(self.ttf1_min, self.ttf1_max))
        self.h2 = HealIndex('F2', np.random.randint(self.ttf2_min, self.ttf2_max))

    def broken(self):
        return self.maintenance != 0

    def __next__(self):

        if self.broken():
            self.maintenance -= 1
            if not self.broken() and self.event_notifier:
                self.event_notifier.notify({'id': str(self), 'time': self.__now(), 'code': 'FIXED', 'severity': 'INFO'})

        if not self.broken():
            if (self.target_speed != 0) or (self.speed > self.speed_threshold):
                try:
                    _, h1 = next(self.h1)
                    _, h2 = next(self.h2)

                    self.speed = (self.speed + (2 - h2) * self.target_speed) / 2
                    self.temperature = (2 - h1) * self.__g(self.temperature, self.ambient_temperature,
                                                           self.temperature_max, self.speed / 1000,
                                                           0.01 * self.speed / 1000)
                    self.pressure = h1 * self.__g(self.pressure, self.ambient_pressure, np.inf,
                                                  self.speed * self.pressure_factor, 0.3 * self.speed / 1000)

                except BrokenMachine as e:
                    self.speed = 0
                    self.temperature = 0
                    self.pressure = 0
                    self.maintenance = -1
                    if self.event_notifier:
                        self.event_notifier.notify({'id': str(self), 'time': self.__now(), 'code': str(e),
                                                    'severity': 'CRITICAL'})
                    raise e

                if self.telemetry_notifier:
                    self.telemetry_notifier.notify(self.get_status())

            else:
                self.speed = 0
                self.temperature = 0
                self.pressure = 0


class Facility(object):

    def __init__(self, temperature=21, pressure=101, cycle_length_min=1, cycle_length_max=5, cycle_duration=60,
                 machines_per_batch=1, operational_speed_min=900, operational_speed_max=1000, clock=None,
                 batch_time=3600, **kwargs):

        self.temperature = temperature
        self.pressure = pressure
        self.cycle_length_min = cycle_length_min
        self.cycle_length_max = cycle_length_max
        self.cycle_duration = cycle_duration
        self.batch_time = batch_time
        self.operational_speed_min = operational_speed_min
        self.operational_speed_max = operational_speed_max
        self.machines_per_batch = machines_per_batch
        self.clock = clock
        self.planning = {}
        self.machines = []
        self.t = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.t % self.batch_time == 0:
            s = [self.machines[i] for i in sorted(np.random.choice(range(len(self.machines)),
                                                                   size=self.machines_per_batch, replace=False))]
            cycle_lengths = np.random.randint(self.cycle_length_min, self.cycle_length_max + 1,
                                              size=self.machines_per_batch)
            durations = cycle_lengths * self.cycle_duration
            starts = self.t + np.array([np.random.randint(0, self.batch_time - d) for d in durations])
            stops = starts + durations - 1
            speeds = np.random.randint(self.operational_speed_min, self.operational_speed_max, self.machines_per_batch)
            planning = [{'start': sta, 'stop': sto, 'speed': spd} for sta, sto, spd in zip(starts, stops, speeds)]
            self.planning = dict(zip(s, planning))
            # print(self.planning)

        for m in self.machines:
            if m in self.planning:
                if self.t == self.planning[m]['start']:
                    m.target_speed = self.planning[m]['speed']
                elif self.t == self.planning[m]['stop']:
                    m.target_speed = 0
            try:
                next(m)
            except BrokenMachine as bm:
                m.repair(self.batch_time - (self.t % self.batch_time))
        self.t += 1

    def add(self, machine):
        machine.facility = self
        machine.clock = self.clock
        machine.ambient_pressure = self.pressure
        machine.ambient_temperature = self.temperature
        self.machines.append(machine)
