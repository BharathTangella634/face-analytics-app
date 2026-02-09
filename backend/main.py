import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import base64
from model_utils import load_all_models, process_frame

app = FastAPI()

# Crucial for deployment: Allows the frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EMOTION_PATH = "./checkpoints/best_model.pt"
AGE_PATH = "./checkpoints/age_model_resnet.pt"
CASCADE_PATH = "./checkpoints/haarcascade_frontalface_default.xml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models once at start
emo_model, age_model, face_cascade = load_all_models(EMOTION_PATH, AGE_PATH, CASCADE_PATH, device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image sent by the browser
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid Image")

        # Process (Drawing boxes/text happens here)
        processed_img, results = process_frame(img, emo_model, age_model, face_cascade, device)

        # Convert back to base64 for the website to show
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "image": f"data:image/jpeg;base64,{img_str}",
            "data": results
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)