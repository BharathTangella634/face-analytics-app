import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
# from model import *
import sys
import torchvision.transforms as T
import torch.nn as nn
import timm
from torchvision.models import resnet, efficientnet_b0

class AgeEstimationModel(nn.Module):

    def __init__(self, input_dim, output_nodes, model_name, pretrain_weights):
        super(AgeEstimationModel, self).__init__()

        self.input_dim = input_dim
        self.output_nodes = output_nodes
        self.pretrain_weights = pretrain_weights

        if model_name == 'resnet':
            self.model = resnet.resnet50(weights=pretrain_weights)
            self.model.fc = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                          nn.Linear(in_features=2048, out_features=256, bias=True),
                                          nn.Linear(in_features=256, out_features=self.output_nodes, bias=True))

        elif model_name == 'efficientnet':
            self.model = efficientnet_b0()
            self.model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                  nn.Linear(in_features=1280, out_features=256, bias=True),
                                                  nn.Linear(in_features=256, out_features=self.output_nodes, bias=True))

        elif model_name == 'vit':
            self.model = timm.create_model('vit_small_patch14_dinov2.lvd142m', img_size=128, pretrained=pretrain_weights)
            
            # num_features = model.blocks[11].mlp.fc2.out_features
            num_features = 384
            self.model.head = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                            nn.Linear(num_features, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, self.output_nodes))

        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, x):
        x = self.model(x)
        return x
    

class Face_Emotion_CNN(nn.Module):
  def __init__(self):
    super(Face_Emotion_CNN, self).__init__()
    self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
    self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
    self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
    self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
    self.cnn6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
    self.cnn7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
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
    x = self.log_softmax(self.fc3(x))
    return x

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Load Models ----------------
def load_trained_model_emotion(model_path):
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model


def load_trained_model_age(model_path):
    model = AgeEstimationModel(
        input_dim=3,
        output_nodes=1,
        model_name='resnet',
        pretrain_weights='IMAGENET1K_V2'
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def FER_live_cam():

    model1 = load_trained_model_emotion(
        r'/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial-Emotion-Recognition-PyTorch-ONNX/PyTorch/best_model.pt'
    )

    model2 = load_trained_model_age(
        r"/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial_Age_estimation_PyTorch/checkpoints/epoch-29-loss_valid-4.61.pt"
    )

    emotion_dict = {
        0: 'Neutral', 1: 'Happiness', 2: 'Surprise',
        3: 'Sadness', 4: 'Anger', 5: 'Disgust', 6: 'Fear'
    }

    # Emotion transform
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507395516207,), (0.255128989415,))
    ])

    # Age transform (moved outside loop)
    age_transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # Webcam (IP stream)
    cap = cv2.VideoCapture("http://172.25.109.99:5000/video")
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    face_cascade = cv2.CascadeClassifier(
        './models/haarcascade_frontalface_default.xml'
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Faster + correct detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # -------- Emotion --------
            face_gray = gray[y:y + h, x:x + w]
            resize_frame = cv2.resize(face_gray, (48, 48)).astype(np.float32)

            X = Image.fromarray(resize_frame)
            X = val_transform(X).unsqueeze(0).to(device)

            with torch.no_grad():
                log_ps = model1(X)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                pred = emotion_dict[int(top_class.cpu().numpy())]

            # -------- Age --------
            face_crop = frame[y:y + h, x:x + w]
            face_pil = Image.fromarray(
                cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            )

            input_data = age_transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model2(input_data)

            age_estimation = outputs.item()

            # -------- Text (unchanged) --------
            pred_text = f"{pred} {age_estimation:.2f}"

            cv2.putText(frame, pred_text, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.putText(frame, pred_text, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    FER_live_cam()
