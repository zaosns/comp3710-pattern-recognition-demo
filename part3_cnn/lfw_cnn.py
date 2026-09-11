"""Train the Part 3.1 convolutional network on LFW faces."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "part3_lfw"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "part3_lfw_cnn.pt"


@dataclass(frozen=True)
class SplitIndices:
    """Indices for mutually exclusive training, validation, and test sets."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


class LFWCNN(nn.Module):
    """CNN matching the Part 3.1 architecture shown in the course slides."""

    def __init__(
        self,
        num_classes: int = 7,
        hidden_features: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if hidden_features < 1:
            raise ValueError("hidden_features must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 9, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (1, 50, 37):
            raise ValueError("expected images with shape (batch, 1, 50, 37)")
        return self.classifier(self.features(images))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a repeatable experiment."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, Apple MPS, or CPU, unless a device is requested."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def stratified_split_indices(
    labels: np.ndarray,
    test_size: float = 0.25,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> SplitIndices:
    """Create stratified splits; validation_fraction is of the training pool."""

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")

    indices = np.arange(labels.size)
    training_pool, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train, validation = train_test_split(
        training_pool,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels[training_pool],
    )
    return SplitIndices(train=train, validation=validation, test=test)


def normalise_from_training(
    images: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Standardise every split using scalar statistics from training images."""

    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError("images must have shape (samples, height, width)")
    if len(train_indices) == 0:
        raise ValueError("the training split cannot be empty")

    training_images = images[train_indices]
    mean = float(training_images.mean(dtype=np.float64))
    standard_deviation = float(training_images.std(dtype=np.float64))
    if standard_deviation <= np.finfo(np.float32).eps:
        raise ValueError("training images must have non-zero variance")

    normalised = (images - mean) / standard_deviation
    return normalised.astype(np.float32, copy=False), mean, standard_deviation


def make_data_loaders(
    images: np.ndarray,
    labels: np.ndarray,
    splits: SplitIndices,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Convert LFW arrays into PyTorch DataLoaders."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    image_tensor = torch.from_numpy(images[:, np.newaxis])
    label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def dataset(indices: np.ndarray) -> TensorDataset:
        index_tensor = torch.from_numpy(np.asarray(indices, dtype=np.int64))
        return TensorDataset(image_tensor[index_tensor], label_tensor[index_tensor])

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset(splits.train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        dataset(splits.validation),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset(splits.test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, validation_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Run one training or evaluation epoch and return aggregate metrics."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += labels.size(0)

    if total_samples == 0:
        raise ValueError("loader cannot be empty")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[list[dict[str, float]], int]:
    """Train the model and restore the checkpoint with best validation accuracy."""

    if epochs < 1:
        raise ValueError("epochs must be positive")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, float]] = []
    best_epoch = 1
    best_validation_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(row)

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:02d}/{epochs}: "
            f"train loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['accuracy']:.2%}; "
            f"validation loss={validation_metrics['loss']:.4f}, "
            f"acc={validation_metrics['accuracy']:.2%}"
        )

    if best_state is None:
        raise RuntimeError("training did not produce a model state")
    model.load_state_dict(best_state)
    return history, best_epoch


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and predictions without retaining gradients."""

    model.eval()
    actual_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            actual_batches.append(labels.numpy())
            prediction_batches.append(logits.argmax(dim=1).cpu().numpy())
    if not actual_batches:
        raise ValueError("loader cannot be empty")
    return np.concatenate(actual_batches), np.concatenate(prediction_batches)


def plot_learning_curves(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in history],
        label="Validation",
    )
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(
        epochs,
        [row["train_accuracy"] for row in history],
        label="Train",
    )
    axes[1].plot(
        epochs,
        [row["validation_accuracy"] for row in history],
        label="Validation",
    )
    axes[1].set(title="Classification accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("Part 3.1 LFW CNN training history")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_confusion(
    actual: np.ndarray,
    predicted: np.ndarray,
    class_names: np.ndarray,
    output_path: Path,
) -> np.ndarray:
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=np.arange(len(class_names)),
        normalize="true",
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title="Part 3.1 CNN on LFW test images",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            colour = "white" if matrix[row, column] > 0.5 else "#0b356d"
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color=colour,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return matrix


def plot_sample_predictions(
    images: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    class_names: np.ndarray,
    output_path: Path,
    count: int = 12,
) -> None:
    selected = representative_prediction_indices(
        actual, predicted, len(class_names), count
    )
    images = images[selected]
    actual = actual[selected]
    predicted = predicted[selected]
    count = len(selected)
    figure, axes = plt.subplots(3, 4, figsize=(11, 9))
    for axis, image, actual_label, predicted_label in zip(
        axes.flat,
        images[:count],
        actual[:count],
        predicted[:count],
        strict=False,
    ):
        axis.imshow(image, cmap="gray")
        correct = actual_label == predicted_label
        axis.set_title(
            f"Pred: {class_names[predicted_label]}\nTrue: {class_names[actual_label]}",
            color="#187a2f" if correct else "#b22222",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes.flat[count:]:
        axis.axis("off")
    figure.suptitle("Representative LFW test predictions (correct and incorrect)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def representative_prediction_indices(
    actual: np.ndarray,
    predicted: np.ndarray,
    num_classes: int,
    count: int = 12,
) -> np.ndarray:
    """Choose class-diverse correct examples, then class-diverse mistakes."""

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("actual and predicted must be matching one-dimensional arrays")
    if count < 1:
        raise ValueError("count must be positive")

    selected: list[int] = []
    for correctly_classified in (True, False):
        for class_index in range(num_classes):
            candidates = np.flatnonzero(
                (actual == class_index)
                & ((actual == predicted) == correctly_classified)
            )
            if candidates.size:
                selected.append(int(candidates[0]))
            if len(selected) == min(count, len(actual)):
                return np.asarray(selected, dtype=np.int64)

    selected_set = set(selected)
    for index in range(len(actual)):
        if index not in selected_set:
            selected.append(index)
        if len(selected) == min(count, len(actual)):
            break
    return np.asarray(selected, dtype=np.int64)


def save_predictions(
    indices: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    class_names: np.ndarray,
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["dataset_index", "actual_label", "predicted_label", "correct"])
        for index, actual_label, predicted_label in zip(
            indices, actual, predicted, strict=True
        ):
            writer.writerow(
                [
                    int(index),
                    class_names[actual_label],
                    class_names[predicted_label],
                    bool(actual_label == predicted_label),
                ]
            )


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    """Load LFW, train the CNN, and write assessment evidence."""

    seed_everything(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    lfw = fetch_lfw_people(
        data_home=args.data_dir,
        min_faces_per_person=args.min_faces,
        resize=args.resize,
    )
    if lfw.images.shape[1:] != (50, 37):
        raise ValueError(f"expected 50x37 LFW images, received {lfw.images.shape[1:]}")

    splits = stratified_split_indices(
        lfw.target,
        test_size=args.test_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    normalised_images, training_mean, training_standard_deviation = (
        normalise_from_training(lfw.images, splits.train)
    )
    train_loader, validation_loader, test_loader = make_data_loaders(
        normalised_images,
        lfw.target,
        splits,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = LFWCNN(
        num_classes=len(lfw.target_names),
        hidden_features=args.hidden_features,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    start_time = time.perf_counter()
    history, best_epoch = train_model(
        model,
        train_loader,
        validation_loader,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    training_seconds = time.perf_counter() - start_time

    actual, predicted = predict(model, test_loader, device)
    test_accuracy = float(np.mean(actual == predicted))
    report_text = classification_report(
        actual,
        predicted,
        target_names=lfw.target_names,
        zero_division=0,
    )
    report = classification_report(
        actual,
        predicted,
        target_names=lfw.target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = plot_confusion(
        actual,
        predicted,
        lfw.target_names,
        args.output_dir / "confusion_matrix.png",
    )
    plot_learning_curves(history, args.output_dir / "learning_curves.png")
    plot_sample_predictions(
        lfw.images[splits.test],
        actual,
        predicted,
        lfw.target_names,
        args.output_dir / "sample_predictions.png",
    )
    save_predictions(
        splits.test,
        actual,
        predicted,
        lfw.target_names,
        args.output_dir / "predictions.csv",
    )
    (args.output_dir / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    torch.save(
        {
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "class_names": lfw.target_names.tolist(),
            "training_mean": training_mean,
            "training_standard_deviation": training_standard_deviation,
            "best_epoch": best_epoch,
            "seed": args.seed,
        },
        args.checkpoint,
    )

    class_counts = np.bincount(lfw.target, minlength=len(lfw.target_names))
    metrics: dict[str, object] = {
        "dataset": "Labeled Faces in the Wild (funneled)",
        "device": str(device),
        "seed": args.seed,
        "samples": int(len(lfw.images)),
        "image_shape": [1, int(lfw.images.shape[1]), int(lfw.images.shape[2])],
        "classes": int(len(lfw.target_names)),
        "class_names": lfw.target_names.tolist(),
        "class_counts": {
            name: int(class_counts[index])
            for index, name in enumerate(lfw.target_names)
        },
        "training_samples": int(len(splits.train)),
        "validation_samples": int(len(splits.validation)),
        "testing_samples": int(len(splits.test)),
        "stratified_splits": True,
        "normalisation": {
            "source": "training split only",
            "mean": training_mean,
            "standard_deviation": training_standard_deviation,
        },
        "architecture": {
            "convolution_blocks": 2,
            "channels": 32,
            "kernel_size": 3,
            "hidden_features": args.hidden_features,
            "dropout": args.dropout,
            "parameters": parameter_count,
        },
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_accuracy": history[best_epoch - 1]["validation_accuracy"],
        "training_seconds": training_seconds,
        "test_accuracy": test_accuracy,
        "test_macro_f1": report["macro avg"]["f1-score"],
        "test_weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report,
        "normalised_confusion_matrix": matrix.tolist(),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Device: {device}; parameters: {parameter_count:,}")
    print(
        f"Split: {len(splits.train)} train / {len(splits.validation)} validation / "
        f"{len(splits.test)} test"
    )
    print(f"Best validation epoch: {best_epoch}; test accuracy: {test_accuracy:.2%}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(report_text)
    print(f"Results written to {args.output_dir.resolve()}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--min-faces", type=int, default=70)
    parser.add_argument("--resize", type=float, default=0.4)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
