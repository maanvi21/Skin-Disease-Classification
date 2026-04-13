import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns

# ====================== CONFIG ======================
DATA_DIR = './data'
BATCH_SIZE = 32
NUM_EPOCHS = 25           
LEARNING_RATE = 0.0003    

if __name__ == '__main__':
    print("=" * 60)
    print("Skin Disease Classification - DenseNet121")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ====================== DATA LOADING ======================
    print("\n=== Loading dataset ===")

    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {train_dataset.classes}")

    # ====================== WEIGHTED RANDOM SAMPLER ======================
    print("\n=== Creating WeightedRandomSampler for imbalance ===")

    train_labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(train_labels)
    class_sample_count = np.array([class_counts[i] for i in range(num_classes)])
    weight_per_class = 1. / (class_sample_count + 1e-8)
    sample_weights = np.array([weight_per_class[label] for label in train_labels])

    sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(sample_weights), 
                                    replacement=True)

    # ====================== MODEL ======================
    print("\n=== Loading DenseNet121 ===")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    # Freeze backbone initially
    for param in model.features.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier.in_features
    # DenseNet classifier usually benefits from dropout in sparse fine-tuning
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes)
    )
    
    model = model.to(device)

    # ====================== LOSS, OPTIMIZER & LOADERS ======================
    class_weights = torch.tensor(weight_per_class, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor=0.5, patience=3)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    print("Using DenseNet121 + WeightedRandomSampler + Weighted Loss")

    # ====================== TRAINING ======================
    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")

    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        # Scheduler step
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_densenet_skin.pth")
            print(f"   → New best accuracy: {best_acc:.2f}% (model saved)")

        # Unfreeze transition block 3 and dense block 4 after epoch 6
        if epoch == 6:
            print("🔓 Unfreezing late blocks of DenseNet121...")
            for name, param in model.named_parameters():
                if "features.denseblock4" in name or "features.transition3" in name or "features.norm5" in name:
                    param.requires_grad = True

            optimizer = optim.Adam([
                {'params': model.classifier.parameters(), 'lr': 0.0001},
                {'params': [p for n, p in model.named_parameters() if ("features.denseblock4" in n or "features.transition3" in n or "features.norm5" in n)], 'lr': 1e-5}
            ], weight_decay=1e-4)

    print(f"\n✅ Training finished! Best Validation Accuracy: {best_acc:.2f}%")
    print("Best model saved as 'best_densenet_skin.pth'")
