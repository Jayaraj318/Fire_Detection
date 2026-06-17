# Fire Detection System

## YOLO-ViT Hybrid with Optical Flow & Explainability

A comprehensive fire detection system using a hybrid YOLO-ViT architecture with optical flow for temporal analysis and Grad-CAM for explainability.

![Architecture](docs/architecture.png)

## Features

- 🔥 **YOLO Branch**: Object detection for localized fire regions
- 👁️ **ViT Branch**: Global scene understanding via Vision Transformer
- 🌊 **Optical Flow**: Temporal motion analysis for flame dynamics
- 🔗 **Multi-Modal Fusion**: Learned attention-based feature combination
- ⏱️ **Temporal Validation**: Requires N consecutive frames to confirm fire
- 📊 **Grad-CAM Explainability**: Visual heatmaps for detection decisions
- 📱 **Alert System**: SMS (Twilio) and Email (SMTP) notifications
- 🖥️ **Web UI**: Modern dashboard for monitoring and detection

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# OR: source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Alerts (Optional)

Edit `configs/alert_config.json`:

```json
{
  "camera_location": "Your Location",
  "email": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "from_address": "your-email@gmail.com",
    "username": "your-email@gmail.com",
    "password": "your-app-password",
    "recipients": ["alert@example.com"]
  },
  "sms": {
    "enabled": true,
    "twilio_account_sid": "your-sid",
    "twilio_auth_token": "your-token",
    "from_number": "+1234567890",
    "to_numbers": ["+0987654321"]
  }
}
```

### 3. Start the Backend

```bash
cd src/api
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

## Project Structure

```
fire-detection-system/
├── data/                    # Dataset storage
│   ├── raw/                 # Raw images
│   ├── processed/           # Processed data
│   └── splits/              # Train/val/test splits
├── models/                  # Model weights
│   ├── yolo/               # YOLO checkpoints
│   ├── checkpoints/        # Full system checkpoints
├── src/                     # Source code
│   ├── detection/          # Core detection modules
│   ├── optical_flow/       # Motion analysis
│   ├── explainability/     # Grad-CAM
│   ├── alert_system/       # SMS/Email alerts
│   ├── api/                # FastAPI backend
│   └── training/           # Training scripts
├── frontend/               # Next.js web UI
├── tests/                  # Test suite
└── configs/                # Configuration files
```

## Training

### Prepare Dataset

1. Download fire detection dataset from Kaggle
2. Organize into `data/raw/images` and `data/raw/labels`
3. Run data splitting:

```bash
python -m src.preprocessing.split_data
```

### Train YOLO

```bash
python -m src.training.train_yolo --epochs 100 --batch 16
```

### Train Full System

```bash
python -m src.training.train --epochs 100 --batch 8 --wandb
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/detect` | POST | Single image detection |
| `/detect/base64` | POST | Base64 image detection |
| `/alerts` | GET | Get alert history |
| `/ws/video` | WebSocket | Real-time video stream |

## Testing

```bash
pytest tests/ -v
```

## Technologies

- **PyTorch**: Deep learning framework
- **Ultralytics YOLOv8**: Object detection
- **Transformers**: Vision Transformer
- **OpenCV**: Optical flow & image processing
- **FastAPI**: Backend API
- **Next.js 14**: Frontend UI
- **Twilio**: SMS alerts
- **SMTP**: Email alerts

## License

MIT License
