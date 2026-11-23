from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an independent crop disease classifier."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "disease",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "crop_disease_model.pt",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def validate_dataset(data_dir: Path) -> None:
    required = [data_dir / "train", data_dir / "val", data_dir / "test"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Disease dataset split directories are missing: " + ", ".join(missing)
        )

    train_classes = {path.name for path in required[0].iterdir() if path.is_dir()}
    for split_path in required[1:]:
        split_classes = {path.name for path in split_path.iterdir() if path.is_dir()}
        if split_classes != train_classes:
            raise ValueError(
                f"{split_path.name} classes do not match train classes."
            )
    if not train_classes:
        raise ValueError("No disease class directories found in the training split")


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output = args.output.resolve()
    validate_dataset(args.data_dir)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        from sklearn.metrics import (
            confusion_matrix,
            precision_recall_fscore_support,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install torch, torchvision, "
            "pillow, and scikit-learn."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        args.data_dir / "train",
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        args.data_dir / "val",
        transform=evaluation_transform,
    )
    test_dataset = datasets.ImageFolder(
        args.data_dir / "test",
        transform=evaluation_transform,
    )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Training and validation class mappings do not match")
    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Training and test class mappings do not match")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    class_names = [
        name
        for name, _ in sorted(
            train_dataset.class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]
    from src.agents.disease_agent import CNNNeuralNet

    model = CNNNeuralNet(3, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.15)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        correct = 0
        val_total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                predicted = model(images).argmax(dim=1)
                correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_accuracy = correct / val_total if val_total else 0.0
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={running_loss / len(train_dataset):.6f} "
            f"val_accuracy={val_accuracy:.6f}"
        )

    model.eval()
    test_targets = []
    test_predictions = []
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            predicted = model(images).argmax(dim=1).cpu().tolist()
            test_predictions.extend(predicted)
            test_targets.extend(labels.tolist())

    precision, recall, f1, support = precision_recall_fscore_support(
        test_targets,
        test_predictions,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            test_targets,
            test_predictions,
            average="macro",
            zero_division=0,
        )
    )
    matrix = confusion_matrix(
        test_targets,
        test_predictions,
        labels=list(range(len(class_names))),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "image_size": args.image_size,
            "normalize_mean": None,
            "normalize_std": None,
            "architecture": "CNN_NeuralNet",
        },
        args.output,
    )

    report = {
        "dataset": str(args.data_dir),
        "test_images": len(test_dataset),
        "class_names": class_names,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
        "model_artifact": str(args.output),
    }
    report_path = args.output.with_suffix(".evaluation.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
