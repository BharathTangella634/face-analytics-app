# AuraSense AI - Face Analytics App

A modern web application for real-time age estimation and emotion recognition using deep learning.

## Features

- 🎭 **Emotion Detection**: Recognizes 7 emotions (Neutral, Happiness, Surprise, Sadness, Anger, Disgust, Fear)
- 👤 **Age Estimation**: Predicts age from facial images
- 📸 **Multiple Input Methods**: Upload images or use live webcam
- ⚡ **Real-time Processing**: Live analysis with instant results
- 🎨 **Modern UI**: Beautiful glass-morphism design with Tailwind CSS

## Project Structure

```
age-emotion-app/
├── api/                          # Vercel serverless functions
│   └── index.py                  # FastAPI app entry point
├── backend/                       # ML models and utilities
│   ├── main.py                   # FastAPI server (local)
│   ├── model_utils.py            # Model loading & inference
│   ├── requirements.txt           # Python dependencies
│   └── checkpoints/              # Pre-trained models
│       ├── best_model.pt         # Emotion detection model
│       ├── age_model_resnet.pt   # Age estimation model
│       └── haarcascade_frontalface_default.xml
├── frontend/                      # Web interface
│   ├── index.html                # Main page
│   └── script.js                 # Client-side logic
├── vercel.json                   # Vercel configuration
├── .vercelignore                 # Files to ignore in deployment
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Deployment to Vercel

### Prerequisites

1. GitHub account
2. Vercel account (free at https://vercel.com)
3. Git installed locally

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Prepare for Vercel deployment"
git push -u origin main
```

### Step 2: Deploy on Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click **"New Project"**
3. Import the GitHub repository
4. Click **"Import"**
5. Configure project:
   - **Framework**: Other
   - **Build Command**: Leave empty (automatic)
6. Click **"Deploy"**

### Step 3: Wait for Deployment

Your app will be live in a few minutes! You'll get a URL like:
`https://your-project-name.vercel.app`

## How It Works

### Frontend
- Built with **Tailwind CSS** + vanilla JavaScript
- Supports image upload via drag-and-drop
- Live webcam analysis with real-time predictions
- Displays detected faces with emotion and age labels

### Backend (API)
- **FastAPI** serverless function
- Accepts image uploads via `/api/predict` endpoint
- Returns:
  - Processed image with bounding boxes
  - Emotion predictions
  - Age estimates
  - Processing latency

### Models

1. **Emotion Recognition**: 7-layer CNN trained on FER-2013 dataset
2. **Age Estimation**: ResNet-50 with custom head

## API Documentation

### POST /api/predict

Upload an image and get predictions.

**Request:**
```
Content-Type: multipart/form-data
Body: file (image file)
```

**Response:**
```json
{
  "success": true,
  "image": "data:image/jpeg;base64,...",
  "data": [
    {
      "emotion": "Happiness",
      "age": 28.5
    }
  ]
}
```

## Local Development

### Setup

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run backend
cd backend
python main.py

# Open frontend
# Serve frontend/index.html on http://localhost:8000
```

### Test

```bash
# Test API with curl
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg"
```

## Technologies

- **Frontend**: HTML5, JavaScript, Tailwind CSS
- **Backend**: Python, FastAPI, PyTorch
- **Models**: CNN, ResNet-50
- **Deployment**: Vercel Serverless Functions
- **Image Processing**: OpenCV, NumPy

## Performance

- **Average Latency**: 100-500ms per image
- **Supported Formats**: JPEG, PNG, WebP
- **Max File Size**: 10MB (typical)

## Troubleshooting

### Models not loading on Vercel?
- Ensure checkpoint files are tracked in Git
- Check file paths are relative
- Monitor Vercel logs in dashboard

### CORS errors?
- Already configured in `/api/index.py`
- All origins are allowed

### Slow predictions?
- Models run on CPU (Vercel free tier limitation)
- GPU deployment available on paid plans

## License

MIT

## Support

For issues and questions, open a GitHub issue.
