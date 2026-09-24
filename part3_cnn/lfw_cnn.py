"""Part 3.1: classify the LFW faces with a small PyTorch CNN."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "part3_lfw"
CHECKPOINT = ROOT / "checkpoints" / "part3_lfw_cnn.pt"


class LFWCNN(nn.Module):
    """CNN with two 3x3 convolution layers."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # First convolutional layer: 1 input channel, 32 filters, 3x3 kernels.
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second convolutional layer: 32 input channels, 32 filters, 3x3 kernels.
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 9, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(X))


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(data_dir: Path, batch_size: int):
    lfw_people = fetch_lfw_people(
        data_home=data_dir,
        min_faces_per_person=70,
        resize=0.4,
    )
    X = lfw_people.images.astype(np.float32)
    Y = lfw_people.target.astype(np.int64)
    target_names = lfw_people.target_names

    print("X_min:", X.min(), "X_max:", X.max())
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42
    )
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]
    print("X_train shape:", X_train.shape)

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader, target_names


def train_epoch(model, loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    # Step 5 - Batch repetition: repeat Steps 1-4 for every batch.
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # Clear the gradients left by the previous batch.
        optimizer.zero_grad()

        # Step 1 - Forward pass: pass the images through the network.
        predictions = model(X)

        # Step 2 - Loss function: compare predictions with the correct labels.
        loss = loss_function(predictions, y)

        # Step 3 - Backpropagation: calculate the gradients of the weights.
        loss.backward()

        # Step 4 - Optimizer: update the model weights using the gradients.
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_correct += (predictions.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


@torch.inference_mode()
def test_model(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    actual, predicted = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        predictions = model(X)
        total_loss += loss_function(predictions, y).item() * len(y)
        actual.extend(y.cpu().numpy())
        predicted.extend(predictions.argmax(1).cpu().numpy())
    accuracy = float(np.mean(np.asarray(actual) == np.asarray(predicted)))
    return total_loss / len(loader.dataset), accuracy, np.asarray(actual), np.asarray(predicted)


def plot_history(history, output_path: Path) -> None:
    epochs = np.arange(1, len(history) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row[0] for row in history])
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, [row[1] for row in history])
    axes[1].set(title="Training accuracy", xlabel="Epoch", ylabel="Accuracy")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = get_device(args.device)
    train_loader, test_loader, target_names = make_loaders(
        args.data_dir, args.batch_size
    )

    model = LFWCNN(len(target_names)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        history.append((train_loss, train_accuracy))
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"loss={train_loss:.4f}, accuracy={train_accuracy:.2%}"
        )

    training_seconds = time.perf_counter() - start
    test_loss, test_accuracy, y_test, predictions = test_model(
        model, test_loader, loss_function, device
    )
    report = classification_report(
        y_test, predictions, target_names=target_names, zero_division=0
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    plot_history(history, args.output_dir / "learning_curves.png")
    (args.output_dir / "classification_report.txt").write_text(report)
    metrics = {
        "device": str(device),
        "epochs": args.epochs,
        "training_seconds": training_seconds,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save(
        {"model_state_dict": model.state_dict(), "target_names": target_names.tolist()},
        args.checkpoint,
    )

    print(f"Device: {device}; test accuracy: {test_accuracy:.2%}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
