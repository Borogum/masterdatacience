import time
import json
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
    mqtt_client = mqtt.Client('test_subscribe', True, None, mqtt.MQTTv31)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.on_subscribe = on_subscribe

    mqtt_client.loop_start()
    print("Connecting to broker at %s:%d" % (broker_ip, broker_port))
    mqtt_client.connect(broker_ip, port=broker_port)

    while not mqtt_client.connected_flag:
        print("In wait loop")
        time.sleep(1)

    mqtt_client.subscribe(topic)

    while True:
        try:
            time.sleep(2)
        except KeyboardInterrupt:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()