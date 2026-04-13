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
cam = None

# Full 22 class names from training data dir
class_names = [
    "Acne", "Actinic_Keratosis", "Benign_tumors", "Bullous", "Candidiasis",
    "DrugEruption", "Eczema", "Infestations_Bites", "Lichen", "Lupus",
    "Moles", "Psoriasis", "Rosacea", "Seborrh_Keratoses", "SkinCancer",
    "Sun_Sunlight_Damage", "Tinea", "Unknown_Normal", "Vascular_Tumors",
    "Vasculitis", "Vitiligo", "Warts"
]

def load_model():
    global model, transform, cam
    
    model_path = 'best_densenet_skin_v2.pth'
    
    # Initialize DenseNet121
    model = models.densenet121(weights=None)
    
    # Match the architecture used in train_densenet.py
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, len(class_names))
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Define transform matching training script
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Initialize Grad-CAM
    # Target the last feature block in DenseNet121
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    print(f"DenseNet121 loaded successfully on {device} with {len(class_names)} classes!")

# Load model initially so it's ready on first request
load_model()

def predict_image(image: Image.Image):
    """Predict + Generate Grad-CAM heatmap"""
    if model is None or transform is None or cam is None:
        load_model()

    # Preprocess image for model (must be 224x224)
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass for prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    # Target the predicted class for Grad-CAM
    targets = [ClassifierOutputTarget(predicted_idx.item())]

    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Convert original image to numpy (0-1 range) to overlay heatmap
    rgb_img = np.array(image.resize((224, 224))) / 255.0

    # Overlay heatmap on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Format result
    result = {
        "prediction": predicted_class,
        "confidence": round(confidence_pct, 2),
        "message": get_advice(predicted_class),
        "heatmap": visualization  # numpy array for later conversion to base64
    }

    return result

def get_advice(prediction: str):
    high_risk = ["SkinCancer", "Actinic_Keratosis", "Melanoma"]
    fungal_infectious = ["Tinea", "Candidiasis", "Infestations_Bites"]
    
    if any(x in prediction for x in high_risk):
        return "⚠️ High risk indicated - Please consult a dermatologist immediately."
    elif any(x in prediction for x in fungal_infectious):
        return "Likely fungal or infectious. Usually treatable with medication. Consult a doctor."
    else:
        return "Low risk indicated. If symptoms persist or worsen, please see a healthcare professional."