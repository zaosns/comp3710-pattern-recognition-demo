"""Train a U-Net for categorical OASIS brain segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

if __package__:
    from .oasis_data import (
        DEFAULT_DATA_SOURCE,
        ROOT,
        SegmentationDataset,
        get_device,
        seed_everything,
    )
else:
    from oasis_data import (  # type: ignore
        DEFAULT_DATA_SOURCE,
        ROOT,
        SegmentationDataset,
        get_device,
        seed_everything,
    )


OUTPUT_DIR = ROOT / "outputs" / "part4_unet"
CHECKPOINT = ROOT / "checkpoints" / "part4_unet.pt"
NUM_CLASSES = 4
CLASS_NAMES = ("label_0", "label_1", "label_2", "label_3")


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images):
        return self.layers(images)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, inputs, skip):
        inputs = self.up(inputs)
        return self.conv(torch.cat((skip, inputs), dim=1))


class UNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, base_channels: int = 32):
        super().__init__()
        channels = base_channels
        self.encoder1 = DoubleConv(1, channels)
        self.encoder2 = DoubleConv(channels, channels * 2)
        self.encoder3 = DoubleConv(channels * 2, channels * 4)
        self.encoder4 = DoubleConv(channels * 4, channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(channels * 8, channels * 16)
        self.decoder4 = UpBlock(channels * 16, channels * 8, channels * 8)
        self.decoder3 = UpBlock(channels * 8, channels * 4, channels * 4)
        self.decoder2 = UpBlock(channels * 4, channels * 2, channels * 2)
        self.decoder1 = UpBlock(channels * 2, channels, channels)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, images):
        encoder1 = self.encoder1(images)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        bottleneck = self.bottleneck(self.pool(encoder4))
        decoder4 = self.decoder4(bottleneck, encoder4)
        decoder3 = self.decoder3(decoder4, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)
        return self.classifier(decoder1)


def soft_dice_loss(logits, one_hot_targets, smooth=1e-5):
    probabilities = torch.softmax(logits, dim=1)
    dimensions = (0, 2, 3)
    intersection = (probabilities * one_hot_targets).sum(dim=dimensions)
    denominator = (probabilities + one_hot_targets).sum(dim=dimensions)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - dice.mean()


def segmentation_loss(logits, one_hot_targets, dice_weight=0.6):
    class_targets = one_hot_targets.argmax(dim=1)
    cross_entropy = F.cross_entropy(logits, class_targets)
    dice = soft_dice_loss(logits, one_hot_targets)
    return (1 - dice_weight) * cross_entropy + dice_weight * dice


def hard_dice_totals(logits, one_hot_targets):
    predictions = F.one_hot(logits.argmax(1), NUM_CLASSES).permute(0, 3, 1, 2)
    dimensions = (0, 2, 3)
    intersection = (predictions * one_hot_targets).sum(dim=dimensions)
    denominator = (predictions + one_hot_targets).sum(dim=dimensions)
    return intersection.float(), denominator.float()


def make_loaders(args):
    loaders = []
    for split in ("train", "validate", "test"):
        dataset = SegmentationDataset(args.data_source, split, args.image_size)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=split == "train",
                num_workers=args.workers,
                pin_memory=torch.cuda.is_available(),
            )
        )
    return loaders


def run_epoch(model, loader, device, dice_weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_images = 0
    intersection = torch.zeros(NUM_CLASSES, device=device)
    denominator = torch.zeros(NUM_CLASSES, device=device)

    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = segmentation_loss(logits, masks, dice_weight)
            if training:
                loss.backward()
                optimizer.step()

        batch_intersection, batch_denominator = hard_dice_totals(logits, masks)
        intersection += batch_intersection
        denominator += batch_denominator
        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)

    dice = (2 * intersection + 1e-5) / (denominator + 1e-5)
    return total_loss / total_images, dice.cpu()


def save_checkpoint(model, epoch, dice, args):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_dice": dice.tolist(),
            "image_size": args.image_size,
            "base_channels": args.base_channels,
        },
        args.checkpoint,
    )


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


@torch.inference_mode()
def plot_predictions(model, loader, device, output_path):
    model.eval()
    images, masks, names = next(iter(loader))
    images = images[:6].to(device)
    targets = masks[:6].argmax(1)
    predictions = model(images).argmax(1).cpu()

    figure, axes = plt.subplots(len(images), 3, figsize=(9, 3 * len(images)))
    for row in range(len(images)):
        axes[row, 0].imshow(images[row, 0].cpu(), cmap="gray")
        axes[row, 1].imshow(targets[row], vmin=0, vmax=3)
        axes[row, 2].imshow(predictions[row], vmin=0, vmax=3)
        axes[row, 0].set_ylabel(names[row])
        for axis in axes[row]:
            axis.axis("off")
    axes[0, 0].set_title("MRI")
    axes[0, 1].set_title("Ground truth")
    axes[0, 2].set_title("Prediction")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def dice_text(dice):
    return ", ".join(
        f"{name}={score:.3f}" for name, score in zip(CLASS_NAMES, dice)
    )


def run(args):
    seed_everything(args.seed)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, test_loader = make_loaders(args)
    model = UNet(NUM_CLASSES, args.base_channels).to(device)
    history = []

    if args.inference_only:
        checkpoint = load_checkpoint(model, args.checkpoint)
        best_epoch = checkpoint["epoch"]
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        best_minimum_dice = 0.0
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, _ = run_epoch(
                model, train_loader, device, args.dice_weight, optimizer
            )
            validation_loss, validation_dice = run_epoch(
                model, validation_loader, device, args.dice_weight
            )
            scheduler.step()
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_dice": validation_dice.tolist(),
                }
            )
            if validation_dice.min().item() > best_minimum_dice:
                best_minimum_dice = validation_dice.min().item()
                best_epoch = epoch
                save_checkpoint(model, epoch, validation_dice, args)
            print(
                f"Epoch {epoch:02d}/{args.epochs}: "
                f"validation loss={validation_loss:.4f}; "
                f"{dice_text(validation_dice)}"
            )
        load_checkpoint(model, args.checkpoint)

    test_loss, test_dice = run_epoch(
        model, test_loader, device, args.dice_weight
    )
    plot_predictions(
        model,
        test_loader,
        device,
        args.output_dir / "segmentation_examples.png",
    )
    metrics = {
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_dice": test_dice.tolist(),
        "all_labels_above_0.9": bool(torch.all(test_dice > 0.9)),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Test DSC: {dice_text(test_dice)}")
    print(f"All labels above 0.9: {metrics['all_labels_above_0.9']}")
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", type=Path, default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dice-weight", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference-only", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
