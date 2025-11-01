# 🧠 Deepfake Detection API (Backend)

This repository contains the backend implementation for **Deepfake Video Detection** using a hybrid **ResNet50 + BiLSTM** deep learning architecture.  
The API is built with **FastAPI** and performs real-time detection of manipulated (fake) and authentic (real) videos.

---

## 🚀 Features
- 🎥 **Video-based Deepfake Detection** using CNN + LSTM architecture  
- 🧩 **Face detection** with MTCNN before frame analysis  
- ⚡ **FastAPI-powered backend** with endpoints for single & batch video prediction  
- 📊 **Confidence scoring** with sigmoid probabilities  
- 🧱 **Supports multiple formats**: `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`

---

## 🧰 Tech Stack
| Component | Technology |
|------------|-------------|
| Deep Learning | PyTorch, TorchVision |
| API Framework | FastAPI, Uvicorn |
| Video & Image Processing | OpenCV, Pillow, facenet-pytorch |
| Deployment | Python 3.10 +, Uvicorn server |

---

## 📂 Folder Structure
```
Deepfake-Detection/
├── backend.py             # FastAPI backend (main entry)
├── requirements.txt       # Dependencies
├── .gitignore             # Ignore cache, env, large files
├── saved_models/          # (optional) Pretrained weights (<100 MB)
└── dataset/               # (optional) Local dataset folder
```

---

## 📦 Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Dipakk7/Deepfake-Detection.git
cd Deepfake-Detection
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # (Windows: venv\Scripts\activate)
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the FastAPI Server
```bash
python backend.py
```

---

## 🎯 API Endpoints
| Method | Endpoint | Description |
|---------|-----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed system health |
| `POST` | `/predict` | Predict deepfake on a single uploaded video |
| `POST` | `/predict/batch` | Analyze multiple videos in one request |
| `GET` | `/info` | Model & configuration details |

### Example (Single Video)
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample_videos/real_sample.mp4"
```

---

## 🧠 Model Weights

Pretrained model (108 MB) is hosted externally due to GitHub’s 100 MB limit.  
👉 **[Download model weights from Google Drive](https://drive.google.com/file/d/1ZSp7lvbaQhoN51nsaO8Oi0WA1cMFYnb6/view?usp=sharing)**  

Alternatively, download via command line:
```bash
pip install gdown
gdown 1ZSp7lvbaQhoN51nsaO8Oi0WA1cMFYnb6 -O saved_models/model_epoch_30.pth
```

After downloading, place it in:
```
Deepfake-Detection/saved_models/model_epoch_30.pth
```
Then set the environment variable:
```bash
export MODEL_PATH="saved_models/model_epoch_30.pth"
```

---

## 📂 Dataset

The training dataset used in this project comes from the **[Deepfake Detection Challenge on Kaggle](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)**.

To download it using the Kaggle API:
```bash
kaggle competitions download -c deepfake-detection-challenge
unzip dfdc_train_part_0.zip -d dataset/
```

Once downloaded and extracted, place the data in your project structure as follows:
```
Deepfake-Detection/
└── dataset/
    ├── dfdc_train_part_0/
    ├── dfdc_train_part_1/
    └── … (other folders)
```

This dataset was released by Meta and partners for the **Deepfake Detection Challenge**, containing thousands of real and fake videos for training and evaluation.

---

## 🧪 Training Overview
The **ResNet50 + BiLSTM** model:
- Uses ResNet50 as a spatial feature extractor  
- Employs a BiLSTM for temporal dependencies across frames  
- Trained with **class-imbalance handling** (weighted loss + oversampling)  
- Achieved **≈ 94–95% accuracy** on validation data  

---

## 📈 Results
| Metric | Value |
|---------|--------|
| Accuracy | 94.3 % |
| Precision | 93.8 % |
| Recall | 94.6 % |
| F1-Score | 94.2 % |

*(Adjust values based on your actual model results.)*

---

## ⚙️ Environment Variables
| Variable | Description | Default |
|-----------|-------------|----------|
| `MODEL_PATH` | Path to model weights | `model_epoch_30.pth` |
| `PREDICTION_THRESHOLD` | Classification threshold | `0.5` |
| `HOST` | Host for FastAPI | `0.0.0.0` |
| `PORT` | Port | `8000` |

---

## 🧾 Requirements
See `requirements.txt` for all dependencies:
```
fastapi==0.115.0
uvicorn==0.30.6
torch
torchvision
facenet-pytorch==2.5.3
opencv-python==4.10.0.84
Pillow==10.2.0
numpy==1.26.4
tqdm==4.66.5
python-multipart==0.0.9
```

---

## 🪪 License
This project is licensed under the **MIT License**.  
Feel free to use and modify with proper credit.

---

## 👨‍💻 Author
**Dipak Khandagale**  
💼 B.Tech AI | Deepfake Detection Researcher | ML Engineer  
📧 khandagaledipak47@gmail.com  
🔗 [GitHub Profile](https://github.com/Dipakk7)

---

**Built with ❤️ using FastAPI, PyTorch, and OpenCV to ensure digital media authenticity.**
