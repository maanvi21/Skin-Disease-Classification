# 🩺 Skin Disease Classification & Intelligence

> Deep learning-based multi-class skin disease classifier comparing multiple architectures (MobileNet, EfficientNet, and DenseNet) on 22 disease categories, fully packaged into a dynamic Flask Web App with Grad-CAM interpretability.

---

## 🏆 Final Results & Architecture Comparison

In this project, we explicitly compared three powerful computer vision architectures. Classifying 22 highly-imbalanced skin diseases is a challenging task (random chance baseline = ~4.5%).

| Architecture | Validation Accuracy | Characteristics |
|--------------|---------------------|-----------------|
| **DenseNet121** (Final Choice) | **62.23%** | Very strong results. Chosen as the final model due to its dense feature maps, which map perfectly with `Grad-CAM` for creating pristine explainable visualizations for the web app. |
| **MobileNetV2** | **62.74%** | Extremely lightweight and fast. It converged quickly, but MobileNet's depthwise separable convolutions made precise pixel-level Grad-CAM explanations slightly less defined than DenseNet. |
| **EfficientNetB2** | **Poor / Did not converge well** | Struggled to converge effectively on this specific imbalanced dataset setup despite its theoretical power. |

> ⚠️ **Note:** Achieving over 62% accuracy across 22 fine-grained dermatological classes significantly outperforms random guessing. This DenseNet model achieved this result with a relatively short training run (**25 initial epochs + 15 fine-tuning epochs**), leaving us highly open to extending the training in the future to push accuracy even further!

![Web App Interface](image.png)

---

## ⚖️ Solving Severe Class Imbalance

Real-world medical datasets suffer heavily from class imbalance (e.g., thousands of benign moles, but very few rare vasculitis cases). If untreated, the model will just predict the majority class. We solved this doubly through sampling and loss manipulations:

### 1. Sampling-Level: `WeightedRandomSampler`
Each training sample is assigned a weight inversely proportional to its class frequency. Rarer classes are mathematically sampled more often during data-loading, forcing the model to look at rare diseases just as often as common ones.

### 2. Loss-Level: Weighted `CrossEntropyLoss`
Class weights are passed directly to the loss function. Even if a minority class sample is seen, its gradient contribution to the learning pipeline is artificially amplified, heavily punishing the model if it misclassifies a rare skin disease.

---

## ⚙️ Core Methodologies

### 1. Fine-Tuning & Unfreezing Strategy
Rather than training from scratch, we load `ImageNet` weights. We freeze the core convolutional network and train just the classifier layer.
For **DenseNet121**, after the classifier plateaued at `~50%` accuracy, we **globally unfroze the entire network** (`requires_grad = True`) and continued training on a micro learning rate (`1e-5`). This aggressive fine-tuning is what pushed the validation accuracy up to `62.23%`.

### 2. Aggressive Data Augmentation
Applied dynamically via `torchvision.transforms` to artificially expand the dataset and prevent memorization:
- Random Horizontal & Vertical Flips (`p=0.5`)
- Aggressive Rotations (`45 degrees`)
- Color Jitter (Brightness/Contrast variance) to simulate different lighting conditions and skin tones.

### 3. Optimizer & Schedulers
- **Optimizer:** Adam
- **Scheduler:** `ReduceLROnPlateau` (halves the LR when validation loss plateaus to enable precise convergence around local minima).

---

## 🌐 The Diagnostic Web App

This repository includes a stunning, completely integrated Flask web application. It bridges the gap between the raw backend PyTorch tensors and the end-user.

### Features
1. **Dynamic Glassmorphism Interface**: A highly responsive drag-and-drop UI written in Vanilla CSS and JS.
2. **Real-time Inference**: Forwards the uploaded image to the DenseNet model and extracts raw confidence probabilities.
3. **Grad-CAM Integration**: Extracts activation gradients from `model.features[-1]` in DenseNet121, highlighting the exact pixels in the image that caused the model to predict the specific disease. 
4. **Intelligent Banners**: If a high-risk lesion (e.g., Melanoma, Squamous Cell Carcinoma) is detected, the UI instantly pulses a high-risk red alert urging immediate medical consultation.

---

## 🚀 How to Run the Project

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Start the Web App Server
```bash
python app.py
# Or if using the virtual environment:
venv\Scripts\python app.py
```
The application will safely spin up on `http://127.0.0.1:5000`. 

### 3. Train Base Models
```bash
python train_densenet.py
# Or for MobileNet
python train_mobilenet.py
```

### 4. Aggressively Fine-Tune / Resume Training
If you want to unfreeze models and squeeze extra accuracy:
```bash
python resume_train_densenet.py
```

---

## 🖥️ Hardware & Structure
- **Trained heavily on:** NVIDIA RTX 4050 Laptop GPU (6GB VRAM)
- Built with standard `torch`, `torchvision`, and `pytorch-grad-cam`.

*Built for educational and research exploration in bridging AI with deep-dermatology workflows.*