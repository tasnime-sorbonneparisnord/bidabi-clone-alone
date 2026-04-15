"""
ResNet-18 training pipeline for image classification.

This module supports:
- image dataset loading from data/train, data/val, data/test or data/raw
- train/validation/test splitting
- data augmentation
- training with ResNet-18
- saving the best model
- metrics export and plot saving
"""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18

try:
    import umap
    umap_available = True
except ImportError:
    umap_available = False

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(filepath):
    return str(filepath).lower().endswith(IMAGE_EXTENSIONS)


def build_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, test_transform


def create_image_datasets(data_dir, image_size, val_ratio, test_ratio, seed):
    data_path = Path(data_dir)
    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test"
    raw_path = data_path / "raw"

    train_transform, test_transform = build_transforms(image_size)

    if train_path.exists() and val_path.exists():
        train_dataset = datasets.ImageFolder(
            root=train_path,
            transform=train_transform,
            is_valid_file=is_image_file,
        )
        val_dataset = datasets.ImageFolder(
            root=val_path,
            transform=test_transform,
            is_valid_file=is_image_file,
        )
        test_dataset = None
        if test_path.exists():
            test_dataset = datasets.ImageFolder(
                root=test_path,
                transform=test_transform,
                is_valid_file=is_image_file,
            )
        classes = train_dataset.classes
    elif raw_path.exists():
        raw_dataset = datasets.ImageFolder(
            root=raw_path,
            transform=train_transform,
            is_valid_file=is_image_file,
        )
        total = len(raw_dataset)
        if total < 3:
            raise ValueError("Dataset too small: au moins 3 images sont requises.")

        val_size = max(1, int(total * val_ratio))
        test_size = max(1, int(total * test_ratio))
        train_size = total - val_size - test_size
        if train_size < 1:
            raise ValueError(
                "Partition train/val/test impossible avec les paramètres actuels."
            )

        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            raw_dataset,
            [train_size, val_size, test_size],
            generator=generator,
        )
        val_dataset.dataset.transform = test_transform
        test_dataset.dataset.transform = test_transform
        classes = raw_dataset.classes
    else:
        raise FileNotFoundError(
            "Aucun jeu de données image trouvé. Créez data/train et data/val, "
            "ou utilisez data/raw avec des sous-dossiers de classes."
        )

    return train_dataset, val_dataset, test_dataset, classes


def build_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, val_loader, test_loader


def create_resnet18(num_classes, pretrained=True):
    model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    for param in model.parameters():
        param.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, num_classes))
    return model


def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    output_path,
    epochs,
    patience,
    device,
):
    best_val_loss = float("inf")
    best_metrics = {}
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.4)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}
            torch.save(
                {
                    "class_names": train_loader.dataset.dataset.classes
                    if hasattr(train_loader.dataset, "dataset")
                    else train_loader.dataset.classes,
                    "state_dict": model.state_dict(),
                },
                output_path,
            )
            print(f"Best model saved to {output_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    return best_metrics, history


def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probabilities.cpu().tolist())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def plot_history(history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(output_dir / "loss_history.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.savefig(output_dir / "accuracy_history.png")
    plt.close()


def plot_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(output_path)
    plt.close()


def plot_roc_curves(labels, probs, classes, output_path):
    y_bin = label_binarize(labels, classes=list(range(len(classes))))
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ResNet-18 training pipeline")
    parser.add_argument("--data-dir", default="data", help="Root image data directory")
    parser.add_argument("--output-dir", default="model", help="Directory for model and reports")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio for data/raw")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio for data/raw")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    train_dataset, val_dataset, test_dataset, classes = create_image_datasets(
        args.data_dir,
        args.image_size,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    train_loader, val_loader, test_loader = build_loaders(
        train_dataset, val_dataset, test_dataset, args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_resnet18(num_classes=len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_model_path = output_path / "best_model_resnet18_finetuned.pth"
    plot_path = output_path / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    best_metrics, training_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        output_path=best_model_path,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
    )

    metrics = {
        "dataset": args.data_dir,
        "classes": classes,
        "best_epoch": best_metrics.get("epoch"),
        "best_val_loss": best_metrics.get("val_loss"),
        "best_val_acc": best_metrics.get("val_acc"),
        "training_history": training_history,
    }

    history_path = output_path / "training_history.json"
    metrics_path = output_path / "metrics.json"
    save_json(metrics, metrics_path)
    save_json(training_history, history_path)
    plot_history(training_history, plot_path)

    if test_loader is not None:
        model_data = torch.load(best_model_path, map_location=device)
        model.load_state_dict(model_data["state_dict"])
        preds, labels, probs = predict_all(model, test_loader, device)
        report = classification_report(labels, preds, target_names=classes, output_dict=True)
        cm = confusion_matrix(labels, preds)
        metrics["test_report"] = report
        save_json(report, output_path / "test_classification_report.json")
        save_json(cm.tolist(), output_path / "confusion_matrix.json")
        plot_confusion_matrix(cm, classes, plot_path / "confusion_matrix.png")
        if len(classes) > 1:
            plot_roc_curves(labels, probs, classes, plot_path / "roc_curves.png")
        save_json(metrics, metrics_path)
        print(classification_report(labels, preds, target_names=classes))
    else:
        print("Aucun ensemble de test trouvé. Le modèle a été entraîné et validé.")

    print(f"Best model path: {best_model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
