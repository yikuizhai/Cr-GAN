"""Linear evaluation and fine-tuning for CR-GAN."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CR-GAN downstream evaluation")
    parser.add_argument("--train-dir", required=True, help="ImageFolder training directory")
    parser.add_argument("--val-dir", required=True, help="ImageFolder validation directory")
    parser.add_argument("--output", default="runs/finetune")
    parser.add_argument("--pretrained", help="SimCLR checkpoint")
    parser.add_argument(
        "--initialization",
        choices=["random", "simclr", "imagenet", "xavier"],
        default="simclr",
    )
    parser.add_argument("--mode", choices=["linear", "finetune"], default="linear")
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.4),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    train_dataset = datasets.ImageFolder(args.train_dir, train_transform)
    val_dataset = datasets.ImageFolder(args.val_dir, val_transform)
    if len(train_dataset.classes) != args.classes:
        raise ValueError(
            f"--classes={args.classes}, but train-dir contains "
            f"{len(train_dataset.classes)} classes"
        )
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train and validation class mappings differ")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    return train_loader, val_loader


def load_simclr_backbone(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    backbone_state = {}
    for key, value in state.items():
        if not key.startswith("backbone."):
            continue
        key = key.removeprefix("backbone.")
        if key.startswith("fc."):
            continue
        backbone_state[key] = value
    if not backbone_state:
        raise ValueError("checkpoint does not contain SimCLR backbone weights")
    result = model.load_state_dict(backbone_state, strict=False)
    unexpected = set(result.unexpected_keys)
    missing = set(result.missing_keys)
    if unexpected or missing != {"fc.weight", "fc.bias"}:
        raise RuntimeError(
            f"unexpected SimCLR state mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def build_model(args: argparse.Namespace) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if args.initialization == "imagenet" else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, args.classes)

    if args.initialization == "simclr":
        if not args.pretrained:
            raise ValueError("--pretrained is required for simclr initialization")
        load_simclr_backbone(model, args.pretrained)
    elif args.initialization == "xavier":
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    if args.mode == "linear":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name in {"fc.weight", "fc.bias"}
    return model


def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
    return total_loss / len(loader.dataset), 100.0 * correct / len(loader.dataset)


def validate(loader: DataLoader, model: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(targets.tolist())
    return {
        "accuracy": 100.0 * accuracy_score(labels, predictions),
        "precision": 100.0 * precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": 100.0 * recall_score(labels, predictions, average="macro", zero_division=0),
        "f1": 100.0 * f1_score(labels, predictions, average="macro", zero_division=0),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = build_loaders(args)
    model = build_model(args).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss().to(device)

    best_metrics: dict[str, float] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            train_loader, model, criterion, optimizer, device
        )
        metrics = validate(val_loader, model, device)
        print(
            f"epoch={epoch} loss={train_loss:.6f} train_acc={train_accuracy:.2f} "
            f"val_acc={metrics['accuracy']:.2f}"
        )
        if best_metrics is None or metrics["accuracy"] > best_metrics["accuracy"]:
            best_metrics = metrics
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "metrics": metrics},
                output / "best_model.pth.tar",
            )

    assert best_metrics is not None
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(best_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
