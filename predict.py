import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = None

# Your 10 class names (make sure this list exactly matches the order used during training)
class_names = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis",
    "Vascular lesion"
    # Add your 10th class name here if missing
]

def load_model():
    global model, transform

    model_path = 'best_mobilenetv2_skin_fine.pth'

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"✅ Model loaded successfully on {device} with {len(class_names)} classes!")

load_model()

def predict_image(image: Image.Image):
    """Predict + Generate Grad-CAM heatmap"""
    if model is None or transform is None:
        load_model()

    # Preprocess image for model
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass for prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    # ================== Grad-CAM Explainability (Fixed) ==================
    target_layers = [model.features[-1]]  # Last convolutional block in MobileNetV2

    cam = GradCAM(model=model, target_layers=target_layers)   # Removed use_cuda

    # Target the predicted class
    targets = [ClassifierOutputTarget(predicted_idx.item())]

    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Convert original image to numpy (0-1 range)
    rgb_img = np.array(image.resize((224, 224))) / 255.0

    # Overlay heatmap on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Advice message
    result = {
        "prediction": predicted_class,
        "confidence": round(confidence_pct, 2),
        "message": get_advice(predicted_class),
        "heatmap": visualization  # numpy array for later conversion to base64
    }

    return result


def get_advice(prediction: str):
    if any(x in prediction for x in ["Melanoma", "Squamous", "Actinic"]):
        return "⚠️ High risk - Please consult a dermatologist immediately."
    elif "Tinea" in prediction or "Candidiasis" in prediction:
        return "Likely fungal infection. Usually treatable with antifungal medication."
    else:
        return "Appears benign/low risk. Still recommended to see a doctor for confirmation."