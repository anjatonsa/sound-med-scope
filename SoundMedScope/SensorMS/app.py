from flask import Flask
import paho.mqtt.client as mqtt
import threading, time, os, base64, json

app = Flask(__name__)

broker_address = "mosquitto"
broker_port = 1883
pub_topic = "sensor/readings"
AUDIO_FOLDER = "/app/audio_files"

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code "+str(rc))

def on_publish(client, userdata, result):
    print("Data published to topic ", pub_topic)

def simulate_sensor_data(client):
    wav_files = []
    for root, dirs, files in os.walk(AUDIO_FOLDER):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))

    if not wav_files:
        print("No audio files found for simulation.")
        return

    print(f"Found {len(wav_files)} wav files.")

    for wav_file in wav_files:
        try:
            with open(wav_file, "rb") as f:
                audio_data = f.read()
                payload = {
                "filename": os.path.basename(wav_file),
                "data": base64.b64encode(audio_data).decode("utf-8")
                }
            client.publish(pub_topic, json.dumps(payload))
            print(f"Sent audio file: {wav_file}")
            time.sleep(1)
        except Exception as e:
            print(f"Error while publishing {wav_file}: {e}")


@app.route('/')
def index():
    return 'Sensor  microservice'


if __name__ == '__main__':

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_publish = on_publish

    client.connect(broker_address, broker_port, 60)
    client.loop_start()

    threading.Thread(target=simulate_sensor_data, args=(client,), daemon=True).start()

    app.run()
