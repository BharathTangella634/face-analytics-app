# import cv2
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import argparse
# import os
# from model import *
# import numpy as np
# import sys

# sys.path.append(r"/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial_Age_estimation_PyTorch")

# from model2 import AgeEstimationModel
# import torchvision.transforms as T

# # python FER_image.py --path='ntr anger.jpg'

# def load_trained_model_emotion(model_path):
#     model = Face_Emotion_CNN()

#     # num_emotions = 7  # Adjust as needed
#     # model = SENet_FER(num_classes=num_emotions)
#     # model = ResNet50(num_classes=7, channels=1)
#     model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
#     return model

# def load_trained_model_age(model_path):
#     model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name='resnet', pretrain_weights='IMAGENET1K_V2').to('cpu')
#     model.load_state_dict(torch.load(model_path))
#     return model

# def FER_image(img_path):

#     model1 = load_trained_model_emotion(r'/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial-Emotion-Recognition-PyTorch-ONNX/PyTorch/best_model.pt')
#     model2 = load_trained_model_age(r"/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial_Age_estimation_PyTorch/checkpoints/epoch-29-loss_valid-4.61.pt")
    
#     emotion_dict = {0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 3: 'Sadness',
#                     4: 'Anger', 5: 'Disguest', 6: 'Fear'}
    
#     # emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
#     #             4: 'sad', 5: 'surprise', 6: 'neutral'}
    
#     val_transform = transforms.Compose([
#         transforms.ToTensor(), 
#         transforms.Normalize((0.507395516207, ),(0.255128989415, ))
#         ])


#     img = cv2.imread(img_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(img)
#     print(faces)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
#         resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48)).astype(np.float32)
        
#         # print(resize_frame)

#         X = resize_frame/255.0
#         X = Image.fromarray((resize_frame))
#         X = val_transform(X).unsqueeze(0)
#         # X = X.repeat(1, 3, 1, 1)
#         # print(X)
#         with torch.no_grad():
#             model1.eval()
#             log_ps = model1.cpu()(X)
#             print(log_ps)
#             ps = torch.exp(log_ps)
#             top_p, top_class = ps.topk(1, dim=1)
#             pred = emotion_dict[int(top_class.numpy())]
#         with torch.no_grad():
#             model2.eval()
#             # image = img[y:y + h, x:x + w].convert('RGB')
#             face_crop = img[y:y + h, x:x + w]  # Extract the region of interest (ROI)

#             # Convert NumPy array to PIL Image
#             face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

#             # Convert to RGB (not necessary here because OpenCV already loads in BGR)
#             image = face_pil.convert('RGB')

#             transform = T.Compose([T.Resize(((128, 128))),
#                                 T.ToTensor(),
#                                 T.Normalize(mean=[0.485, 0.456, 0.406], 
#                                             std=[0.229, 0.224, 0.225])
#                                 ])
#             input_data = transform(image).unsqueeze(0).to('cpu') 
#             outputs = model2.cpu()(input_data)  # Forward pass through the model
        
#         # Extract the age estimation value from the output tensor
#         age_estimation = outputs.item()
#         pred = pred + " " +  f"{age_estimation:.2f}"
#         # cv2.putText(img, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
#         cv2.putText(img, pred, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3, cv2.LINE_AA)  # Shadow effect
#         cv2.putText(img, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)  # Main text in yellow


#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.grid(False)
#     plt.axis('off')
#     plt.show()



# if __name__ == "__main__":

#     ap = argparse.ArgumentParser()
#     ap.add_argument("-p", "--path", required=True,
#         help="path of image")
#     args = vars(ap.parse_args())
    
#     if not os.path.isfile(args['path']):
#         print('The image path does not exists!!')
#     else:
#         print(args['path'])
#         FER_image(args['path'])



import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
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


# ------------------ Load Emotion Model ------------------
def load_trained_model_emotion(model_path, device):
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model


# ------------------ Load Age Model ------------------
def load_trained_model_age(model_path, device):
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


# ------------------ Main Function ------------------
def FER_image(img_path, model1, model2, device):

    emotion_dict = {
        0: 'Neutral', 1: 'Happiness', 2: 'Surprise',
        3: 'Sadness', 4: 'Anger', 5: 'Disguest', 6: 'Fear'
    }

    # Emotion transform (48x48 grayscale)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507395516207,), (0.255128989415,))
    ])

    # Age transform (128x128 RGB)
    age_transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    img = cv2.imread(img_path)
    if img is None:
        print("Error reading image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use grayscale for detection (better performance)
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ---------------- Emotion Prediction ----------------
        face_gray = gray[y:y + h, x:x + w]
        resize_frame = cv2.resize(face_gray, (48, 48)).astype(np.float32)

        X = Image.fromarray(resize_frame)
        X = val_transform(X).unsqueeze(0).to(device)

        with torch.no_grad():
            log_ps = model1(X)
            ps = torch.exp(log_ps)  # since last layer is log_softmax
            top_p, top_class = ps.topk(1, dim=1)
            emotion_pred = emotion_dict[int(top_class.cpu().numpy())]

        # ---------------- Age Prediction ----------------
        face_crop = img[y:y + h, x:x + w]
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        image_rgb = face_pil.convert('RGB')

        input_data = age_transform(image_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model2(input_data)

        age_estimation = outputs.item()

        # ---------------- Final Text ----------------
        # ---------------- Final Text ----------------
        # ---------------- Final Text ----------------
        # ---------------- Final Text ----------------

        # pred_text = f"{emotion_pred} {age_estimation:.2f}"
        # cv2.putText(img, pred_text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        # cv2.putText(img, pred_text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


        final_text = f"{emotion_pred} | Age: {age_estimation:.1f}"

        img_h, img_w = img.shape[:2]

        # 1️⃣ Base font scale relative to FACE size (not image)
        font_scale = h / 150.0   # main scaling factor
        font_scale = max(0.6, min(font_scale, 2.0))  # clamp for stability

        thickness = max(1, int(font_scale * 2))

        # 2️⃣ Measure text size
        (text_w, text_h), baseline = cv2.getTextSize(
            final_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        # 3️⃣ If text wider than face → shrink until it fits
        while text_w > w and font_scale > 0.5:
            font_scale -= 0.05
            thickness = max(1, int(font_scale * 2))
            (text_w, text_h), baseline = cv2.getTextSize(
                final_text,
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
            final_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA
        )



    # Show final image
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ------------------ Main ------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="path of image")
    args = vars(ap.parse_args())

    if not os.path.isfile(args['path']):
        print('The image path does not exist!!')
    else:
        print("Processing:", args['path'])

        # Load models once
        emotion_model = load_trained_model_emotion(
            r'/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial-Emotion-Recognition-PyTorch-ONNX/PyTorch/best_model.pt',
            device
        )

        age_model = load_trained_model_age(
            r"/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial_Age_estimation_PyTorch/checkpoints/epoch-29-loss_valid-4.61.pt",
            device
        )

        FER_image(args['path'], emotion_model, age_model, device)
