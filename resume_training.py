import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from collections import Counter

# ====================== CONFIG ======================
DATA_DIR = './data'
BATCH_SIZE = 32
NUM_EPOCHS = 15          # Additional epochs you want to train
LEARNING_RATE_HEAD = 0.0003
LEARNING_RATE_BACKBONE = 1e-5

if __name__ == '__main__':
    print("=" * 60)
    print("Resuming Training - MobileNetV2 Skin Disease")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # ====================== DATA ======================
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
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

    # Weighted Sampler
    train_labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    weight_per_class = 1. / (class_counts + 1e-8)
    sample_weights = np.array([weight_per_class[label] for label in train_labels])

    sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(sample_weights), 
                                    replacement=True)

    # ====================== LOAD PRETRAINED MODEL ======================
    print("\n=== Loading best saved model ===")
    
    model = models.mobilenet_v2(weights=None)  # Start without default weights
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Load your saved best weights
    model.load_state_dict(torch.load("best_mobilenet_skin.pth", weights_only=True))
    model = model.to(device)

    print("Successfully loaded best model weights!")

    # Unfreeze the backbone (we assume it's already partially unfrozen from previous training)
    for param in model.features[-6:].parameters():   # Unfreeze last 6 blocks
        param.requires_grad = True

    # ====================== LOSS & OPTIMIZER ======================
    class_weights = torch.tensor(weight_per_class, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Use different learning rates
    optimizer = optim.Adam([
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE_HEAD},
        {'params': model.features.parameters(), 'lr': LEARNING_RATE_BACKBONE}
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor=0.5, patience=3)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # ====================== CONTINUE TRAINING ======================
    print(f"\nContinuing training for {NUM_EPOCHS} more epochs...\n")

    best_acc = 51.23  # Your previous best

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

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_mobilenet_skin.pth")
            print(f"   → New best accuracy: {best_acc:.2f}% (model saved)")

    print(f"\n✅ Additional training finished! Final Best Accuracy: {best_acc:.2f}%")