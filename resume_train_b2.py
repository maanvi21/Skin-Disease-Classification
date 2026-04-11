import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = './data'
BATCH_SIZE  = 24
NUM_EPOCHS  = 40
PATIENCE    = 12

LR_HEAD     = 8e-6
LR_BACKBONE = 3e-7

CHECKPOINT  = "best_efficientnet_b2.pth"
SAVE_PATH   = "best_efficientnet_b2.pth"
START_ACC   = 47.74   # ← your current best

# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss — focuses training on hard/misclassified examples.
    Much better than plain CrossEntropy for visually similar classes
    like Acne vs Rosacea vs DrugEruption.
    gamma=2.0 is standard; alpha = class weights for imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 75)
    print("EfficientNet-B2 — Resume Training (Focal Loss + Label Smoothing)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Transforms ────────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15),
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
    print(f"Classes      : {num_classes}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples  : {len(val_dataset):,}\n")

    # ── Weighted sampler ──────────────────────────────────────────────────────
    train_labels     = [label for _, label in train_dataset.samples]
    class_counts     = np.bincount(train_labels, minlength=num_classes)
    weight_per_class = 1.0 / (class_counts + 1e-8)
    sample_weights   = np.array([weight_per_class[l] for l in train_labels])

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
    print(f"Loading checkpoint: {CHECKPOINT}  (best so far = {START_ACC}%)")
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes)

    # Handle both old format (state_dict only) and new format (full checkpoint)
    checkpoint = torch.load(CHECKPOINT, weights_only=False, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        saved_acc = checkpoint.get('best_acc', START_ACC)
        print(f"Full checkpoint loaded. Saved accuracy: {saved_acc:.2f}%")
        best_acc = saved_acc
    else:
        model.load_state_dict(checkpoint)
        best_acc = START_ACC
        print(f"State dict loaded. Using START_ACC: {best_acc:.2f}%")

    model = model.to(device)
    print("Model ready ✓\n")

    # ── Unfreeze last 3 blocks to start ───────────────────────────────────────
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")

    # ── Focal Loss with class weights + label smoothing ───────────────────────
    class_weights_tensor = torch.tensor(
        weight_per_class, dtype=torch.float32).to(device)
    criterion = FocalLoss(
        alpha=class_weights_tensor,
        gamma=2.0,
        label_smoothing=0.1
    )
    print("Loss: FocalLoss(gamma=2.0, label_smoothing=0.1) ✓\n")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.Adam([
        {'params': model.classifier.parameters(),
         'lr': LR_HEAD, 'weight_decay': 1e-4},
        {'params': model.features[-3:].parameters(),
         'lr': LR_BACKBONE, 'weight_decay': 5e-5},
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-9)

    # ── Training loop ─────────────────────────────────────────────────────────
    counter    = 0
    history    = []
    unlocked_5 = False

    print(f"Resuming from {best_acc:.2f}% | "
          f"Epochs: {NUM_EPOCHS} | Patience: {PATIENCE}\n")
    print("-" * 70)

    for epoch in range(NUM_EPOCHS):

        # Progressive unfreeze at epoch 10
        if epoch == 9 and not unlocked_5:
            unlocked_5 = True
            print("\n🔓 Epoch 10: Unlocking 2 more backbone blocks (→ last 5)...")
            for param in model.features[-5:].parameters():
                param.requires_grad = True
            optimizer.add_param_group({
                'params': [p for p in model.features[-5:-3].parameters()
                           if p.requires_grad],
                'lr': LR_BACKBONE * 0.3,
                'weight_decay': 5e-5
            })
            print("   Optimizer updated.\n")

        # ── Train ──
        model.train()
        running_loss  = 0.0
        correct_train = 0
        total_train   = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
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
                outputs      = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total       += labels.size(0)
                correct     += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        history.append({'epoch': epoch+1, 'loss': avg_loss,
                        'train_acc': train_acc, 'val_acc': val_acc})

        print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | "
              f"Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:.2f}% | "
              f"Val: {val_acc:.2f}% | "
              f"LR: {lr_now:.2e}")

        # ── Save best ──
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch,
            }, SAVE_PATH)
            print(f"   ✅ New best: {best_acc:.2f}% — saved to {SAVE_PATH}")
            counter = 0
        else:
            counter += 1
            print(f"   No improvement ({val_acc - best_acc:.2f}%) "
                  f"— patience {counter}/{PATIENCE}")
            if counter >= PATIENCE:
                print("\nEarly stopping triggered.")
                break

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Done. Best validation accuracy: {best_acc:.2f}%")
    print("=" * 70)

    top5 = sorted(history, key=lambda x: x['val_acc'], reverse=True)[:5]
    print("\nTop 5 epochs:")
    for r in top5:
        print(f"  Epoch {r['epoch']:2d} | "
              f"Val: {r['val_acc']:.2f}% | "
              f"Train: {r['train_acc']:.2f}% | "
              f"Loss: {r['loss']:.4f}")