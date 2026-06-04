# Traffic Boy - Backend

This is the Flask backend API for the Traffic Monitoring System. It serves a YOLOv8 model for traffic detection.

## Prerequisites
- Python 3.8+

## Setup Instructions

### 1. Create and Activate a Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Model Weights
Make sure to place your trained YOLO weights (e.g., `best.pt`) inside this root directory.

### 4. Run the Backend Server
```bash
python app.py
```
The server will start on `http://localhost:5000`.

## Important Note on Git and Large Files
This repository uses a `.gitignore` to prevent uploading large model weight files (`*.pt`), virtual environments (`venv`), and sensitive environment variables (`.env`). If you clone this repository on another machine, you will need to provide your own model weights (`best.pt`) and `.env` file locally.
