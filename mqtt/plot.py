import time
import json
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.connected_flag=True
        print("Connected OK")
    else:
        print("Bad connection Returned code=", rc)


def on_message(client, userdata, msg):
    userdata['y'].append(json.loads(msg.payload)['data'])
    userdata['line'].set_ydata(userdata['y'])
    plt.draw()


if __name__ == '__main__':

    # data
    n_values = 20
    x = np.arange(0, n_values)
    y = deque([None] * n_values, maxlen=n_values)

    # mqtt
    broker_ip = 'localhost'
    broker_port = 1883
    topic = 'test'
    mqtt.Client.connected_flag = False
    client = mqtt.Client(client_id='test_plot', clean_session=True, protocol=mqtt.MQTTv31, transport='tcp')
    client.on_connect = on_connect
    client.on_message = on_message

    # graph
    fig, ax = plt.subplots()
    ax.set_ylim(-3, 3)
    ax.set_xlim(0, n_values - 1)
    line, = ax.plot(x, [np.nan] * len(x))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.0f}s'.format(n_values - x - 1)))
    ax.set_xlabel('Seconds ago')

    # Add user data
    client.user_data_set({'line': line, 'y': y})

    client.loop_start()
    print("Connecting to broker at %s:%d" % (broker_ip, broker_port))
    client.connect(broker_ip, port=broker_port)

    while not client.connected_flag:
        print("In wait loop")
        time.sleep(1)

    client.subscribe(topic)
    plt.show()