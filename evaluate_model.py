import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    print("Evaluating best_densenet_skin_v2.pth on Validation Dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = './data'
    val_dir = os.path.join(DATA_DIR, 'val')

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_classes = len(val_dataset.classes)
    
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes)
    )

    model.load_state_dict(torch.load("best_densenet_skin_v2.pth", map_location=device))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Final Validation Accuracy: {acc:.2f}%")
