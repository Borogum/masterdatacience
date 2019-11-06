import time
import json
import numpy as np
import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.connected_flag=True
        print("Connected OK")
    else:
        print("Bad connection Returned code=", rc)


def on_message(client, userdata, msg):
    print("Received: ", json.loads(msg.payload)['data'])


def on_subscribe(client, userdata, mid, granted_qos):
    print('Suscribed')


if __name__ == '__main__':

    broker_ip = 'localhost'
    broker_port = 1883
    topic = 'test'
    mqtt.Client.connected_flag = False
    client = mqtt.Client('test_subscribe', True, None, mqtt.MQTTv31)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_subscribe = on_subscribe

    client.loop_start()
    print("Connecting to broker at %s:%d" % (broker_ip, broker_port))
    client.connect(broker_ip, port=broker_port)

    while not client.connected_flag:
        print("In wait loop")
        time.sleep(1)

    client.subscribe(topic)

    while True:
        try:
            time.sleep(2)
        except KeyboardInterrupt:
            client.loop_stop()
            client.disconnect()