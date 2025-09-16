import os,json, datetime, time, base64
from flask import Flask, jsonify
import paho.mqtt.client as mqtt
import psycopg2
import boto3
from botocore.exceptions import ClientError


app = Flask(__name__)

broker_address = "mosquitto"
broker_port = 1883
sub_topic = "sensor/readings"

pg_host = os.getenv("POSTGRES_HOST", None)
pg_port = os.getenv("POSTGRES_PORT", None)
pg_db = os.getenv("POSTGRES_DB", None)
pg_user = os.getenv("POSTGRES_USER", None)
pg_password = os.getenv("POSTGRES_PASSWORD", None)

minio_endpoint = os.getenv("MINIO_ENDPOINT")
minio_access = os.getenv("MINIO_ACCESS_KEY")
minio_secret = os.getenv("MINIO_SECRET_KEY")
minio_bucket = os.getenv("MINIO_BUCKET")

s3_client = boto3.client(
    "s3",
    endpoint_url=f"http://{minio_endpoint}",
    aws_access_key_id=minio_access,
    aws_secret_access_key=minio_secret,
    region_name="us-east-1"
)

def ensure_bucket():
    try:
        s3_client.head_bucket(Bucket=minio_bucket)
    except ClientError:
        s3_client.create_bucket(Bucket=minio_bucket)
        print(f"Created bucket: {minio_bucket}")

def get_pg_conn(retries=5, delay=3):
    for i in range(retries):
        try:
            conn = psycopg2.connect(
                host=pg_host,
                port=pg_port,
                dbname=pg_db,
                user=pg_user,
                password=pg_password,
            )
            return conn
        except psycopg2.OperationalError as e:
            print(f"Postgres not ready, retrying in {delay}s... ({i+1}/{retries})")
            time.sleep(delay)
    raise Exception("Could not connect to Postgres after several retries")

def ensure_table():
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stethoscope_recordings (
            id SERIAL PRIMARY KEY,
            filename TEXT,
            received_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("stethoscope_recordings table exists.")

def write_metadata(filename):
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO stethoscope_recordings (filename, received_at) VALUES (%s, %s) RETURNING id;",
        (filename, datetime.datetime.now(datetime.timezone.utc))
    )
    rec_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return rec_id

def write_data(filename, file_data):
    s3_client.put_object(Bucket=minio_bucket, Key=filename, Body=file_data)
 
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code " + str(rc))
    client.subscribe(sub_topic, qos=0)
    print("Subcribed to topic " + sub_topic)

def on_message(client, userdata, msg):
    message_data = json.loads(msg.payload.decode())
    print(f"Received message from topic {msg.topic}")
    process_messages(message_data)

def process_messages(msg):
    filename = msg['filename']
    file_data=base64.b64decode(msg['data'])
    try:                  
            #writing data to Postgres
            write_metadata(filename)
            print("Data successfully written to Postgres")

            #writing data to MinIO
            write_data(filename, file_data)
            print("Data successfully written to MinIO")

    except Exception as e:
            print(f"Error storing data: {e}")


@app.route('/stethoscope/readings', methods=['GET'])
def get_all():  
    print("/stethoscope/readings is called.")
    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, filename, received_at FROM stethoscope_recordings ORDER BY received_at DESC;")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        all_recordings = [
            {"id": row[0], "filename": row[1], "received_at": row[2].isoformat()}
            for row in rows
        ]
        return jsonify({"all_stethoscope_readings": all_recordings}), 200
    except Exception as e:
        return jsonify({"error": e}), 500

@app.route('/stethoscope/file/<path:filename>', methods=['GET'])
def get_file(filename):  
    print("/get_by_id is called.")
    try:
        file_obj = s3_client.get_object(Bucket=minio_bucket, Key=filename)
        file_data = file_obj['Body'].read()
        return file_data, 200, {'Content-Type': 'audio/wav'}
    except ClientError as e:
        return jsonify({"error": str(e)}), 404

@app.route('/')
def index():
    return 'Storage microservice'

if __name__ == '__main__':

    #create table and bucket if they dont exist
    ensure_table()
    ensure_bucket()

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker_address, broker_port, 60)
    client.loop_start()

    app.run(host="0.0.0.0", port=5002)
