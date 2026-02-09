import streamlit as st
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
from torchvision.models import resnet, efficientnet_b0
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AuraSense AI | Age & Emotion",
    page_icon="🎭",
    layout="wide"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4A90E2;
        color: white;
    }
    .status-box {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL CLASSES (As provided in your snippets) ---

class AgeEstimationModel(nn.Module):
    def __init__(self, input_dim, output_nodes, model_name, pretrain_weights):
        super(AgeEstimationModel, self).__init__()
        if model_name == 'resnet':
            self.model = resnet.resnet50(weights=None)
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(2048, 256),
                nn.Linear(256, output_nodes)
            )
    def forward(self, x):
        return self.model(x)

class Face_Emotion_CNN(nn.Module):
    def __init__(self):
        super(Face_Emotion_CNN, self).__init__()
        self.cnn1 = nn.Conv2d(1, 8, 3)
        self.cnn2 = nn.Conv2d(8, 16, 3)
        self.cnn3 = nn.Conv2d(16, 32, 3)
        self.cnn4 = nn.Conv2d(32, 64, 3)
        self.cnn5 = nn.Conv2d(64, 128, 3)
        self.cnn6 = nn.Conv2d(128, 256, 3)
        self.cnn7 = nn.Conv2d(256, 256, 3)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.cnn1_bn = nn.BatchNorm2d(8)
        self.cnn2_bn = nn.BatchNorm2d(16)
        self.cnn3_bn = nn.BatchNorm2d(32)
        self.cnn4_bn = nn.BatchNorm2d(64)
        self.cnn5_bn = nn.BatchNorm2d(128)
        self.cnn6_bn = nn.BatchNorm2d(256)
        self.cnn7_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)
        self.dropout = nn.Dropout(0.3)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.pool1(self.cnn1_bn(self.cnn1(x))))
        x = self.relu(self.pool1(self.cnn2_bn(self.dropout(self.cnn2(x)))))
        x = self.relu(self.pool1(self.cnn3_bn(self.cnn3(x))))
        x = self.relu(self.pool1(self.cnn4_bn(self.dropout(self.cnn4(x)))))
        x = self.relu(self.pool2(self.cnn5_bn(self.cnn5(x))))
        x = self.relu(self.pool2(self.cnn6_bn(self.dropout(self.cnn6(x)))))
        x = self.relu(self.pool2(self.cnn7_bn(self.dropout(self.cnn7(x)))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        return self.log_softmax(self.fc3(x))

# --- HELPERS: LOADING MODELS ---

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths (Update these to match your 'models/' folder)
    emotion_path = './models/best_model.pt'
    age_path = './models/age_model.pt'
    cascade_path = './models/haarcascade_frontalface_default.xml'
    
    # Load Emotion
    emotion_model = Face_Emotion_CNN()
    if os.path.exists(emotion_path):
        emotion_model.load_state_dict(torch.load(emotion_path, map_location=device), strict=False)
    emotion_model.to(device).eval()
    
    # Load Age
    age_model = AgeEstimationModel(3, 1, 'resnet', None)
    if os.path.exists(age_path):
        age_model.load_state_dict(torch.load(age_path, map_location=device))
    age_model.to(device).eval()
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    return emotion_model, age_model, face_cascade, device

# --- IMAGE PROCESSING LOGIC ---

def process_frame(frame, emotion_model, age_model, face_cascade, device):
    emotion_dict = {0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness', 4: 'Anger', 5: 'Disgust', 6: 'Fear'}
    
    # Transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507395516207,), (0.255128989415,))
    ])
    age_transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    results = []
    for (x, y, w, h) in faces:
        # Emotion
        face_gray = gray[y:y+h, x:x+w]
        face_gray_resized = cv2.resize(face_gray, (48, 48)).astype(np.float32)
        X_emo = Image.fromarray(face_gray_resized)
        X_emo = val_transform(X_emo).unsqueeze(0).to(device)
        
        with torch.no_grad():
            log_ps = emotion_model(X_emo)
            emotion_idx = int(torch.exp(log_ps).argmax(dim=1).cpu().numpy())
            emotion_pred = emotion_dict.get(emotion_idx, "Unknown")

        # Age
        face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        X_age = age_transform(Image.fromarray(face_rgb)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            age_pred = age_model(X_age).item()

        # Draw UI on Frame
        label = f"{emotion_pred}, ~{int(age_pred)}yrs"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (74, 144, 226), 3)
        cv2.rectangle(frame, (x, y-35), (x+w, y), (74, 144, 226), -1)
        cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        results.append({"Emotion": emotion_pred, "Age": round(age_pred, 1)})
        
    return frame, results

# --- MAIN APP UI ---

def main():
    st.title("🎭 AuraSense AI")
    st.subheader("Real-time Facial Emotion & Age Estimation")
    
    # Load resources
    with st.spinner("Waking up AI models..."):
        emotion_model, age_model, face_cascade, device = load_models()

    # Sidebar
    st.sidebar.title("Settings")
    mode = st.sidebar.selectbox("Choose Input Mode", ["Image Upload", "Live Webcam"])
    st.sidebar.markdown("---")
    st.sidebar.info("This app uses a ResNet backbone for age and a custom CNN for emotions.")

    if mode == "Image Upload":
        uploaded_file = st.file_uploader("Choose a clear face image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert uploaded file to OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Process
            processed_img, data = process_frame(image.copy(), emotion_model, age_model, face_cascade, device)
            
            # Layout for results
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(processed_img, channels="BGR", use_column_width=True, caption="Detected Results")
            with col2:
                st.write("### Analysis Results")
                if data:
                    for i, res in enumerate(data):
                        st.markdown(f"""
                        <div class="status-box">
                            <strong>Face {i+1}</strong><br>
                            Emotion: {res['Emotion']}<br>
                            Estimated Age: {res['Age']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No faces detected.")

    elif mode == "Live Webcam":
        st.write("### Live Stream")
        img_file_buffer = st.camera_input("Capture a moment to analyze")
        
        if img_file_buffer is not None:
            # Simple approach: capture frame and process
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            processed_img, data = process_frame(cv2_img, emotion_model, age_model, face_cascade, device)
            st.image(processed_img, channels="BGR", use_column_width=True)
            
            if data:
                cols = st.columns(len(data))
                for i, res in enumerate(data):
                    cols[i].metric(f"Person {i+1}", res['Emotion'], f"{res['Age']} years")

if __name__ == "__main__":
    main()