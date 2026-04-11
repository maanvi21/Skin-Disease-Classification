import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import os
from collections import Counter

# ================== TRAINING FUNCTION ==================
def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=10, phase="head"):
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{epochs} - {phase} {'='*20}")
        
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss/len(val_loader):.4f}   | Val Acc:   {val_acc:.2f}%")
        
        if scheduler is not None:
            scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_mobilenetv2_skin_{phase}.pth')
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
    
    print(f"🏆 Best Validation Accuracy in {phase} phase: {best_acc:.2f}%")


# ================== MAIN CODE ==================
if __name__ == '__main__':
    # ================== CONFIG ==================
    data_dir = r'C:\Users\maanv\Desktop\skin\Skin-Disease-Classification\data'   # ←←← CHANGE THIS TO YOUR ACTUAL PATH
    batch_size = 16
    num_epochs_head = 10
    num_epochs_fine = 20
    lr_head = 1e-3
    lr_fine = 1e-4
    num_classes = 9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ================== TRANSFORMS ==================
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ================== DATASETS ==================
    train_root = os.path.join(data_dir, 'train')
    val_root   = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(root=train_root, transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=val_root, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    class_names = train_dataset.classes
    print(f"✅ Classes found ({len(class_names)}): {class_names}")

    if len(class_names) != num_classes:
        print(f"❌ ERROR: Found {len(class_names)} classes, expected {num_classes}")
        print("Check your train folder - make sure all 10 class folders have images.")
        exit()

    # ================== CLASS WEIGHTS ==================
    targets = [label for _, label in train_dataset.samples]
    class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    print("Class weights:", {class_names[i]: round(float(w), 3) for i, w in enumerate(class_weights)})

    # ================== MODEL: MobileNetV2 (FIXED) ==================
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)   # ← Fixed here
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ================== STAGE 1: Train classifier head ==================
    print("\n🔥 STAGE 1: Training classifier head...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.classifier.parameters(), lr=lr_head)
    train_phase(model, train_loader, val_loader, criterion, optimizer, epochs=num_epochs_head, phase="head")

    # ================== STAGE 2: Fine-tuning ==================
    print("\n🔥 STAGE 2: Fine-tuning the whole model...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=lr_fine, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=num_epochs_fine, phase="fine")

    print("\n🎉 Training completed! Best model saved as 'best_mobilenetv2_skin_fine.pth'")