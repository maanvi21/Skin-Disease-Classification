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
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5  # Extremely low learning rate for aggressive fine-tuning
OLD_CHECKPOINT = "best_densenet_skin_v2.pth"
NEW_CHECKPOINT = "best_densenet_skin_v3.pth"

if __name__ == '__main__':
    print("=" * 60)
    print("Resuming Skin Disease Classification - DenseNet121 Fully Unfrozen")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ====================== DATA LOADING ======================
    print("\n=== Loading dataset ===")
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')

    # Enhanced Data Augmentation since we are unfreezing entirely
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),  # Increased rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
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
    print(f"Found {num_classes} classes.")

    # ====================== WEIGHTED RANDOM SAMPLER ======================
    train_labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(train_labels)
    class_sample_count = np.array([class_counts[i] for i in range(num_classes)])
    weight_per_class = 1. / (class_sample_count + 1e-8)
    sample_weights = np.array([weight_per_class[label] for label in train_labels])

    sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(sample_weights), 
                                    replacement=True)

    # ====================== MODEL ======================
    print(f"\n=== Loading Pre-Trained Weights from {OLD_CHECKPOINT} ===")
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes)
    )

    try:
        model.load_state_dict(torch.load(OLD_CHECKPOINT, map_location=device))
        print("Models weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        exit()

    # UNFREEZE ALL LAYERS FOR MAXIMUM FINE-TUNING
    for param in model.parameters():
        param.requires_grad = True

    model = model.to(device)

    # ====================== LOSS & OPTIMIZER ======================
    class_weights = torch.tensor(weight_per_class, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # We optimize the ENTIRE model now
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print("\nStarting deep fine-tuning for {} epochs...".format(NUM_EPOCHS))
    
    # We set best_acc to 0.0 so we save at least one check point regardless
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

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

        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), NEW_CHECKPOINT)
            print(f"   New best accuracy: {best_acc:.2f}% (Model saved to {NEW_CHECKPOINT})")

    print(f"\nFine-tuning finished! Best Validation Accuracy achieved: {best_acc:.2f}%")
