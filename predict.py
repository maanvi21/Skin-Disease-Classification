import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = None

# Your exact 10 class names (MUST match training order)
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
]  
# If you have 10 classes and one is missing above, add it here in the same order as your folders

def load_model():
    global model, transform

    model_path = 'best_mobilenetv2_skin_fine.pth'

    # Load MobileNetV2 architecture
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))

    # Load your trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Same transform used during validation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"✅ Model loaded successfully on {device} with {len(class_names)} classes!")

# Load the model when this file is imported
load_model()

def predict_image(image: Image.Image):
    """Predict skin disease from PIL Image"""
    if model is None or transform is None:
        load_model()

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    # Return nice result
    result = {
        "prediction": predicted_class,
        "confidence": round(confidence_pct, 2),
        "message": get_advice(predicted_class)
    }

    return result


def get_advice(prediction):
    """Simple advice based on prediction"""
    if "Melanoma" in prediction or "Squamous" in prediction or "Actinic" in prediction:
        return "⚠️ High risk - Please consult a dermatologist immediately."
    elif "Tinea" in prediction or "Candidiasis" in prediction:
        return "Fungal infection - Usually treatable with antifungal medication."
    else:
        return "Benign / Low risk condition. Still recommended to see a doctor for confirmation."