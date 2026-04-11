# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models, transforms, datasets
# from torch.utils.data import DataLoader, WeightedRandomSampler
# import numpy as np
# from collections import Counter

# DATA_DIR = './data'
# BATCH_SIZE = 24          # Smaller batch size due to larger model
# NUM_EPOCHS = 25
# PATIENCE = 8

# LR_HEAD = 0.0005
# LR_BACKBONE = 1e-5

# if __name__ == '__main__':
#     print("=" * 75)
#     print("EfficientNet-B2 Skin Disease Classification - Initial Training")
#     print("=" * 75)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Stronger augmentations suitable for skin images
#     train_transform = transforms.Compose([
#         transforms.Resize((288, 288)),                    # EfficientNet-B2 prefers ~288
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomVerticalFlip(p=0.3),
#         transforms.RandomRotation(30),
#         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
#         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
#         transforms.ToTensor(),
#         transforms.RandomErasing(p=0.25),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])

#     val_transform = transforms.Compose([
#         transforms.Resize((288, 288)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])

#     train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
#     val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)

#     num_classes = len(train_dataset.classes)
#     print(f"Found {num_classes} classes")

#     # Weighted sampler for imbalance
#     train_labels = [label for _, label in train_dataset.samples]
#     class_counts = np.bincount(train_labels, minlength=num_classes)
#     weight_per_class = 1. / (class_counts + 1e-8)
#     sample_weights = np.array([weight_per_class[label] for label in train_labels])

#     sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

#     # Load EfficientNet-B2
#     model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)

#     # Replace classifier
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

#     # Freeze backbone initially
#     for param in model.features.parameters():
#         param.requires_grad = False

#     model = model.to(device)

#     class_weights = torch.tensor(weight_per_class, dtype=torch.float32).to(device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)

#     optimizer = optim.Adam(model.classifier.parameters(), lr=LR_HEAD, weight_decay=1e-4)

#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

#     best_acc = 0.0
#     counter = 0

#     print(f"Starting initial training for up to {NUM_EPOCHS} epochs...\n")

#     for epoch in range(NUM_EPOCHS):
#         model.train()
#         running_loss = 0.0

#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         avg_train_loss = running_loss / len(train_loader)

#         # Validation
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         val_acc = 100 * correct / total
#         scheduler.step(val_acc)

#         print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")

#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(model.state_dict(), "best_efficientnet_b2.pth")
#             print(f"   → New best: {best_acc:.2f}% (saved)")
#             counter = 0
#         else:
#             counter += 1

#         # Gradual unfreezing after epoch 8
#         if epoch == 8:
#             print("🔓 Unfreezing last 6 blocks of EfficientNet-B2...")
#             for param in model.features[-6:].parameters():
#                 param.requires_grad = True

#             optimizer = optim.Adam([
#                 {'params': model.classifier.parameters(), 'lr': 0.0001},
#                 {'params': model.features.parameters(), 'lr': 5e-6}
#             ])

#         if counter >= PATIENCE:
#             print("Early stopping triggered.")
#             break

#     print(f"\nInitial training finished! Best Accuracy: {best_acc:.2f}%")
#     print("Best model saved as 'best_efficientnet_b2.pth'")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

DATA_DIR = './data'
BATCH_SIZE = 24
NUM_EPOCHS = 20           # How many more epochs you want
PATIENCE = 8

# Gentle learning rates for resume
LR_HEAD = 5e-5
LR_BACKBONE = 8e-7

if __name__ == '__main__':
    print("=" * 80)
    print("EfficientNet-B2 Skin Disease - Resume Training (from epoch 14)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms (same as initial training)
    train_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)

    num_classes = len(train_dataset.classes)

    # Weighted sampler
    train_labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    weight_per_class = 1. / (class_counts + 1e-8)
    sample_weights = np.array([weight_per_class[label] for label in train_labels])

    sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(sample_weights), 
                                    replacement=True)

    # Load the best saved model
    print("\nLoading best model (44.xx%)...")
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("best_efficientnet_b2.pth", weights_only=True, map_location=device))
    model = model.to(device)

    # Unfreeze last 6 blocks (gentle fine-tuning)
    print("Unfreezing last 6 blocks...")
    for param in model.features[-6:].parameters():
        param.requires_grad = True

    class_weights = torch.tensor(weight_per_class, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam([
        {'params': model.classifier.parameters(), 'lr': LR_HEAD, 'weight_decay': 1e-4},
        {'params': model.features.parameters(), 'lr': LR_BACKBONE, 'weight_decay': 5e-5}
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=0, pin_memory=True)

    # <<< IMPORTANT: UPDATE THIS WITH YOUR ACTUAL BEST ACCURACY >>>
    best_acc = 44.37          # ← Change this to the exact "New best" value you saw (e.g. 44.85)

    counter = 0

    print(f"Resuming training for up to {NUM_EPOCHS} more epochs...\n")

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
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_efficientnet_b2.pth")
            print(f"   → New best: {best_acc:.2f}% (saved)")
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nResume training finished. Final best accuracy: {best_acc:.2f}%")