import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet

# --- Model Architectures ---

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

# --- Inference Logic ---

def load_all_models(emotion_path, age_path, cascade_path, device):
    # Load Emotion
    emo_model = Face_Emotion_CNN()
    emo_model.load_state_dict(torch.load(emotion_path, map_location=device), strict=False)
    emo_model.to(device).eval()
    
    # Load Age
    age_model = AgeEstimationModel(3, 1, 'resnet', None)
    age_model.load_state_dict(torch.load(age_path, map_location=device))
    age_model.to(device).eval()
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    return emo_model, age_model, face_cascade

def process_frame(img, emo_model, age_model, face_cascade, device):
    emotion_dict = {0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness', 4: 'Anger', 5: 'Disgust', 6: 'Fear'}
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507395516207,), (0.255128989415,))
    ])
    
    age_transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    results_data = []

    for (x, y, w, h) in faces:
        # Emotion Prediction
        face_gray = gray[y:y+h, x:x+w]
        face_gray = cv2.resize(face_gray, (48, 48)).astype(np.float32)
        emo_input = val_transform(Image.fromarray(face_gray)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            log_ps = emo_model(emo_input)
            ps = torch.exp(log_ps)
            top_class = ps.argmax(dim=1).item()
            emotion_pred = emotion_dict[top_class]

        # Age Prediction
        face_crop = img[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        age_input = age_transform(Image.fromarray(face_rgb)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            age_pred = age_model(age_input).item()

        # Draw on image
        label = f"{emotion_pred} | Age: {age_pred:.1f}"
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        img_h, img_w = img.shape[:2]

        # 1️⃣ Base font scale relative to FACE size (not image)
        font_scale = h / 150.0   # main scaling factor
        font_scale = max(0.6, min(font_scale, 2.0))  # clamp for stability

        thickness = max(1, int(font_scale * 2))

        # 2️⃣ Measure text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        # 3️⃣ If text wider than face → shrink until it fits
        while text_w > w and font_scale > 0.5:
            font_scale -= 0.05
            thickness = max(1, int(font_scale * 2))
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )

        # 4️⃣ Center text horizontally above face
        text_x = x + (w - text_w) // 2
        text_y = y - 10

        # 5️⃣ If going above image → move below face
        if text_y - text_h < 0:
            text_y = y + h + text_h + 5

        # 6️⃣ Clamp inside image boundaries
        text_x = max(0, min(text_x, img_w - text_w))
        text_y = max(text_h, min(text_y, img_h - 5))

        # 7️⃣ Draw background box
        cv2.rectangle(
            img,
            (text_x - 6, text_y - text_h - 6),
            (text_x + text_w + 6, text_y + 6),
            (0, 0, 0),
            -1
        )

        # 8️⃣ Draw text (sharp + anti-aliased)
        cv2.putText(
            img,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        
        results_data.append({"emotion": emotion_pred, "age": round(age_pred, 1)})

    return img, results_data