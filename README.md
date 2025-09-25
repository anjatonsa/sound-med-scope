# 🩺 SoundMedScope MVP

This project is a microservice-based MVP for collecting, storing, and analyzing stethoscope audio recordings.  
It demonstrates an end-to-end pipeline for healthcare AI applications, using modern cloud-native tools.

---

## 🚀 Features

- **MQTT ingestion** – Receives audio data from IoT devices (stethoscopes) - Simulated with a **Sensor Microservice**.  
- **Storage Microservice**  
  - Saves audio files in MinIO (object storage).  
  - Stores metadata (filename, timestamp) in PostgreSQL.  
  - Provides REST endpoints to fetch metadata from PostgreSQL and audiofiles from MinIO.  
- **AI Microservice**  
  - Runs ML model (ONNX) to classify lung sounds (Asthma, COPD, Pneumonia, etc.).  
  - Exposes a `/predict` endpoint returning a prediction for a specified audiofile.  
- **API Gateway**  
  - Single entrypoint for clients(doctors).  
  - Forwards requests to Storage MS and AI MS.  
  - Provides unified REST API for external consumers.  

---
## 📦 Tech Stack

- **Backend:** Python (Flask)  
- **Messaging:** MQTT (Eclipse Mosquitto)  
- **Storage:** MinIO (S3-compatible), PostreSQL 
- **AI/ML:** ONNX Runtime, Librosa, scikit-learn  
- **Containerization:** Docker 

---

## 🔧 Setup & Installation

### Clone repository
```bash
git clone https://github.com/anjatonsa/sound-med-scope.git
cd SoundMedScope
docker-compose up --build
```
### API Endpoints
- GET http://localhost:5003/stethoscope/readings → List all recordings metadata
- GET http://localhost:5003/stethoscope/file/<filename> → Download WAV file
- POST http://localhost:5003/predict { filename: "filename.wav"} → returns prediction





