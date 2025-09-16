from flask import Flask, request, jsonify
import requests,os

app = Flask(__name__)

STORAGE_MS_URL = os.getenv('STORAGE_MS_URL')
AI_MS_URL = os.getenv('AI_MS_URL')

@app.route('/stethoscope/readings', methods=['GET'])
def get_all_readings():
    print("/stethoscope/readings is called.")
    try:
        resp = requests.get(f"{STORAGE_MS_URL}/stethoscope/readings")
        return (resp.content, resp.status_code, resp.headers.items())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stethoscope/file/<path:filename>', methods=['GET'])
def get_file(filename):
    print("/stethoscope/file is called.")
    try:
        resp = requests.get(f"{STORAGE_MS_URL}/stethoscope/file/{filename}", stream=True)
        return (resp.content, resp.status_code, resp.headers.items())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_anomaly():
    print("/predict is called.")

    data = request.json
    if not data:
        print("No data provided")
        return jsonify({"error": "No data provided"}), 500
    
    filename = data['filename']

    resp = requests.post(f"{AI_MS_URL}/predict", json={"filename": filename})
    resp.raise_for_status() 
    prediction = resp.json()

    return jsonify(prediction), resp.status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
