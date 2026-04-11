import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = './data'
BATCH_SIZE  = 24
NUM_EPOCHS  = 30
PATIENCE    = 10

# Much lower LRs — model is already trained, we're polishing
LR_HEAD     = 1e-5     # was 5e-5 → too high, caused val drop after epoch 6
LR_BACKBONE = 5e-7     # was 1e-6 → still too high for resume

CHECKPOINT  = "best_efficientnet_b2.pth"
SAVE_PATH   = "best_efficientnet_b2.pth"
START_ACC   = 47.74    # ← update this to your actual best saved accuracy

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 75)
    print("EfficientNet-B2 — Resume Training (stabilized)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Transforms ────────────────────────────────────────────────────────────
    # Slightly reduced augmentation on resume — model needs stability, not variety
    train_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),             # reduced from 25
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05              # reduced from 0.08
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15),          # reduced from 0.2
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'val'), transform=val_transform)

    num_classes = len(train_dataset.classes)
    print(f"Classes: {num_classes}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples  : {len(val_dataset):,}\n")

    # ── Weighted sampler ──────────────────────────────────────────────────────
    train_labels    = [label for _, label in train_dataset.samples]
    class_counts    = np.bincount(train_labels, minlength=num_classes)
    weight_per_class = 1.0 / (class_counts + 1e-8)
    sample_weights  = np.array([weight_per_class[l] for l in train_labels])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {CHECKPOINT}")
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(
        CHECKPOINT, weights_only=True, map_location=device))
    model = model.to(device)
    print("Checkpoint loaded ✓\n")

    # ── Unfreeze strategy: only last 3 blocks (was 6 — too much) ─────────────
    # Unfreeze fewer blocks = less overfitting, more stable gradients
    for param in model.features.parameters():
        param.requires_grad = False          # freeze everything first

    for param in model.features[-3:].parameters():
        param.requires_grad = True           # unfreeze only last 3 blocks

    for param in model.classifier.parameters():
        param.requires_grad = True           # always train head

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)\n")

    # ── Loss ──────────────────────────────────────────────────────────────────
    class_weights_tensor = torch.tensor(
        weight_per_class, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # ── Optimizer with gradient clipping ──────────────────────────────────────
    optimizer = optim.Adam([
        {'params': model.classifier.parameters(),
         'lr': LR_HEAD, 'weight_decay': 1e-4},
        {'params': model.features[-3:].parameters(),
         'lr': LR_BACKBONE, 'weight_decay': 5e-5},
    ])

    # Cosine annealing — smoother than ReduceLROnPlateau for fine-tuning
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-8)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_acc = START_ACC
    counter  = 0
    history  = []

    print(f"Starting from best_acc = {best_acc:.2f}%")
    print(f"Training for up to {NUM_EPOCHS} epochs "
          f"(early stop patience={PATIENCE})\n")
    print("-" * 65)

    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train   = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping — prevents occasional loss spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss  += loss.item()
            _, preds       = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train   += labels.size(0)

        avg_loss  = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # ── Validate ──
        model.eval()
        correct = 0
        total   = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs        = model(inputs)
                _, predicted   = torch.max(outputs, 1)
                total         += labels.size(0)
                correct       += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        scheduler.step()

        current_lr_head = optimizer.param_groups[0]['lr']
        history.append({'epoch': epoch+1, 'loss': avg_loss,
                        'train_acc': train_acc, 'val_acc': val_acc})

        print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | "
              f"Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:.2f}% | "
              f"Val: {val_acc:.2f}% | "
              f"LR: {current_lr_head:.2e}")

        # ── Save best ──
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"   ✅ New best: {best_acc:.2f}% saved → {SAVE_PATH}")
            counter = 0
        else:
            counter += 1
            gap = val_acc - best_acc
            print(f"   No improvement ({gap:.2f}%) — patience {counter}/{PATIENCE}")
            if counter >= PATIENCE:
                print("\nEarly stopping triggered.")
                break

        # ── Progressive unfreezing: unlock 2 more blocks at epoch 8 ──────────
        if epoch == 7:
            print("\n🔓 Epoch 8: Unlocking 2 more backbone blocks...")
            for param in model.features[-5:].parameters():
                param.requires_grad = True
            # Add new params to optimizer
            optimizer.add_param_group({
                'params': [p for p in model.features[-5:-3].parameters()
                           if p.requires_grad],
                'lr': LR_BACKBONE * 0.5,
                'weight_decay': 5e-5
            })
            print("   Optimizer updated with new params.\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {SAVE_PATH}")
    print("=" * 65)

    # Print top 5 epochs by val acc
    top5 = sorted(history, key=lambda x: x['val_acc'], reverse=True)[:5]
    print("\nTop 5 epochs:")
    for r in top5:
        print(f"  Epoch {r['epoch']:2d} | Val: {r['val_acc']:.2f}% | "
              f"Train: {r['train_acc']:.2f}% | Loss: {r['loss']:.4f}")