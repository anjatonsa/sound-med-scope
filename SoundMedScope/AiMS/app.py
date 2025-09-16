from flask import Flask, request, jsonify
import json,os,requests, io
import numpy as np
import librosa
import onnxruntime
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

STORAGE_MS_URL = os.getenv('STORAGE_MS_URL')

onnx_model_path = '/app/model/best_model.onnx'             
categories = ['Bronchial', 'Pneumonia', 'Asthma', 'Healthy', 'COPD']  # klase za dekodiranje predikcije

def pad_audio(audio, target_len=2048):
    if len(audio) < target_len:
        padding = target_len - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    return audio

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)                         
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)                           
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)             
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)                  
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spec_centroid, axis=1),
        np.mean(spec_bw, axis=1)
    ])
    return features

def run_model(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)      
        y = pad_audio(y)                               
        features = extract_features(y, sr)    
        features = features.reshape(1, -1, 1).astype(np.float32) 

        session = onnxruntime.InferenceSession(onnx_model_path)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: features})
        predicted_index = int(np.argmax(output[0], axis=1)[0])

        encoder = LabelEncoder()
        encoder.fit(categories)
        predicted_class = encoder.inverse_transform([predicted_index])[0]
        print(f"\n Predicted class for '{audio_file}': {predicted_class}")
        return predicted_class
    
    except Exception as e:
        print(f"Error during model inference: {e}")
        return None

def get_file(filename):
    try:
        print(STORAGE_MS_URL)
        resp = requests.get(f"{STORAGE_MS_URL}/stethoscope/file/{filename}", stream=True)
        resp.raise_for_status()
        file = resp.content

    except Exception as e:
        return None
    return file

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({"error": "No filename provided"}), 400

    filename = data['filename']
    try:
        wav_bytes = get_file(filename)
        prediction_result = run_model(io.BytesIO(wav_bytes))
        return jsonify(prediction_result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return 'AI microservice'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)