import time
import numpy as np
import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.connected_flag=True
        print("Connected OK")
    else:
        print("Bad connection Returned code=", rc)


if __name__ == '__main__':

    broker_ip = 'localhost'
    broker_port = 1883
    topic = 'test'
    mqtt.Client.connected_flag = False
    client = mqtt.Client('test_publish', True, None, mqtt.MQTTv31)
    client.on_connect = on_connect
    client.loop_start()
    print("Connecting to broker at %s:%d" % (broker_ip, broker_port))
    client.connect(broker_ip, port=broker_port)

    while not client.connected_flag:
        print("In wait loop")
        time.sleep(1)

    while True:
        payload = '{"data": %.2f}' % np.random.normal()
        client.publish(topic, payload=payload)
        print("Published:",  payload)
        time.sleep(1)

    client.loop_stop()
    client.disconnect()