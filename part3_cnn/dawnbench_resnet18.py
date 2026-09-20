"""Part 3.2: train a from-scratch ResNet-18 on CIFAR-10."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "part3_cifar10"
CHECKPOINT = ROOT / "checkpoints" / "part3_cifar10_resnet18.pt"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class BasicBlock(nn.Module):
    """The two-convolution residual block used by ResNet-18."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(X)
        X = self.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))
        return self.relu(X + residual)


class ResNet18(nn.Module):
    """ResNet-18 with a CIFAR-sized 3x3 input convolution."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(64, 64, 2, 1)
        self.layer2 = self.make_layer(64, 128, 2, 2)
        self.layer3 = self.make_layer(128, 256, 2, 2)
        self.layer4 = self.make_layer(256, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.initialise_parameters()

    @staticmethod
    def make_layer(in_channels, out_channels, blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        layers += [BasicBlock(out_channels, out_channels) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def initialise_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.relu(self.bn1(self.conv1(X)))
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.avgpool(X)
        return self.fc(torch.flatten(X, 1))


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def make_loaders(data_dir, batch_size, workers, download):
    normalise = transforms.Normalize(MEAN, STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalise,
            transforms.RandomErasing(p=0.1),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalise])
    train_set = datasets.CIFAR10(data_dir, train=True, transform=train_transform, download=download)
    validation_set = datasets.CIFAR10(
        data_dir, train=True, transform=test_transform, download=False
    )
    test_set = datasets.CIFAR10(
        data_dir, train=False, transform=test_transform, download=download
    )

    indices = np.random.default_rng(42).permutation(len(train_set))
    validation_size = int(0.05 * len(indices))
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        Subset(train_set, train_indices),
        shuffle=True,
        drop_last=True,
        **options,
    )
    validation_loader = DataLoader(
        Subset(validation_set, validation_indices), shuffle=False, **options
    )
    test_loader = DataLoader(test_set, shuffle=False, **options)
    return train_loader, validation_loader, test_loader


def mixup(X, y, alpha=0.2):
    amount = np.random.beta(alpha, alpha)
    order = torch.randperm(len(X), device=X.device)
    return amount * X + (1 - amount) * X[order], y, y[order], amount


def train_epoch(
    model, loader, loss_function, optimizer, scheduler, scaler, device, use_amp
):
    model.train()
    total_loss = total_correct = total = 0.0
    synchronise(device)
    start = time.perf_counter()

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        X, y1, y2, amount = mixup(X, y)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            predictions = model(X)
            loss = amount * loss_function(predictions, y1)
            loss += (1 - amount) * loss_function(predictions, y2)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        predicted = predictions.argmax(1)
        total_loss += loss.item() * len(y)
        total_correct += amount * (predicted == y1).sum().item()
        total_correct += (1 - amount) * (predicted == y2).sum().item()
        total += len(y)

    synchronise(device)
    seconds = time.perf_counter() - start
    return total_loss / total, total_correct / total, seconds


@torch.inference_mode()
def evaluate(model, loader, loss_function, device, use_amp):
    model.eval()
    total_loss = total_correct = total = 0
    synchronise(device)
    start = time.perf_counter()
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            predictions = model(X)
            loss = loss_function(predictions, y)
        total_loss += loss.item() * len(y)
        total_correct += (predictions.argmax(1) == y).sum().item()
        total += len(y)
    synchronise(device)
    seconds = time.perf_counter() - start
    return total_loss / total, total_correct / total, seconds


def save_checkpoint(model, path: Path, epoch: int, accuracy: float) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_accuracy": accuracy,
        },
        path,
    )


def load_checkpoint(model, path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def train_model(model, loaders, device, use_amp, args):
    train_loader, validation_loader, test_loader = loaders
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.4, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.4,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=10,
        final_div_factor=1000,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_accuracy = -1.0
    history = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, train_seconds = train_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
        )
        validation_loss, validation_accuracy, _ = evaluate(
            model, validation_loader, loss_function, device, use_amp
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "train_seconds": train_seconds,
            }
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save_checkpoint(model, args.checkpoint, epoch, validation_accuracy)
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train accuracy={train_accuracy:.2%}; "
            f"validation accuracy={validation_accuracy:.2%}; {train_seconds:.1f}s"
        )

    load_checkpoint(model, args.checkpoint)
    test_loss, test_accuracy, inference_seconds = evaluate(
        model, test_loader, loss_function, device, use_amp
    )
    total_seconds = time.perf_counter() - start
    metrics = {
        "model": "from-scratch CIFAR-10 ResNet-18",
        "device": str(device),
        "mixed_precision": use_amp,
        "epochs": args.epochs,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "inference_seconds": inference_seconds,
        "total_seconds": total_seconds,
        "above_90_percent": test_accuracy > 0.90,
        "at_least_94_percent": test_accuracy >= 0.94,
        "within_360_seconds": total_seconds <= 360,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"Test accuracy: {test_accuracy:.2%}; total time: {total_seconds:.1f}s; "
        f"inference: {inference_seconds:.2f}s"
    )
    return metrics


def run(args: argparse.Namespace):
    seed_everything()
    device = get_device(args.device)
    use_amp = device.type == "cuda" and args.amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    loaders = make_loaders(
        args.data_dir, args.batch_size, args.workers, args.download
    )
    model = ResNet18().to(device)

    if args.inference_only:
        checkpoint = load_checkpoint(model, args.checkpoint)
        loss_function = nn.CrossEntropyLoss()
        loss, accuracy, seconds = evaluate(
            model, loaders[2], loss_function, device, use_amp
        )
        print(
            f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}; "
            f"loss={loss:.4f}; accuracy={accuracy:.2%}; inference={seconds:.2f}s"
        )
        return

    train_model(model, loaders, device, use_amp, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.set_defaults(amp=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
