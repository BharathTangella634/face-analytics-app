import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
import base64
from backend.model_utils import load_all_models, process_frame

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EMOTION_PATH = os.path.join(os.path.dirname(__file__), "../backend/checkpoints/best_model.pt")
AGE_PATH = os.path.join(os.path.dirname(__file__), "../backend/checkpoints/age_model_resnet.pt")
CASCADE_PATH = os.path.join(os.path.dirname(__file__), "../backend/checkpoints/haarcascade_frontalface_default.xml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models once at startup
try:
    emo_model, age_model, face_cascade = load_all_models(EMOTION_PATH, AGE_PATH, CASCADE_PATH, device)
except Exception as e:
    print(f"Error loading models: {e}")
    emo_model = age_model = face_cascade = None

@app.get("/")
async def root():
    return {"status": "API is running", "service": "AuraSense AI"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if emo_model is None or age_model is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Read the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid Image")

        # Process frame
        processed_img, results = process_frame(img, emo_model, age_model, face_cascade, device)

        # Convert to base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({
            "success": True,
            "image": f"data:image/jpeg;base64,{img_str}",
            "data": results
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=400)

# Vercel Serverless Function Handler
from mangum import Asgi

handler = Asgi(app)
