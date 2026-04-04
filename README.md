# 🧠 Skin Disease Classification with Explainable AI

A deep learning-based web application that classifies skin diseases from images using **MobileNetV2 (PyTorch)** and provides **visual explanations using Grad-CAM**.

---

## 🚀 Features

- 🔍 Skin Disease Classification (9 classes)
- ⚡ Transfer Learning with MobileNetV2
- 🎯 Fine-tuned model for better accuracy
- 🧠 Explainable AI using Grad-CAM
- 🌐 Flask API for real-time predictions
- 🖼️ Heatmap visualization of affected regions

---

## 🏗️ Tech Stack

- **Backend:** Flask  
- **Deep Learning:** PyTorch, Torchvision  
- **Model:** MobileNetV2  
- **Explainability:** Grad-CAM  
- **Image Processing:** OpenCV, PIL  
- **Data Handling:** NumPy, Pandas  

---

## 🧪 Classes

- Actinic keratosis  
- Atopic Dermatitis  
- Benign keratosis  
- Dermatofibroma  
- Melanocytic nevus  
- Melanoma  
- Squamous cell carcinoma  
- Tinea / Ringworm / Candidiasis  
- Vascular lesion  

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/skin-disease-classification.git
cd skin-disease-classification
2. Create virtual environment (PowerShell)
python -m venv venv
.\venv\Scripts\Activate
```
### 3. Install dependencies
pip install -r requirements.txt

### 🏋️‍♂️ Model Training
python train.py

### Training Strategy
Stage 1: Train classifier head
Stage 2: Fine-tune entire model
Uses data augmentation, class balancing, and LR scheduling

### 🔮 Run Prediction (Flask API)
python app.py

### 📸 How It Works
-Upload an image
-Model predicts disease class
-Grad-CAM generates heatmap
-Heatmap highlights important regions

🧠 Grad-CAM Explainability
Grad-CAM helps:
-Visualize important regions in the image
-Improve trust in predictions
-Make the model interpretable

### 📊 Sample Output
```
{
  "prediction": "Melanoma",
  "confidence": 92.34,
  "message": "⚠️ High risk - Please consult a dermatologist immediately."
}
```
