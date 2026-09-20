"""Train and visualise a convolutional VAE on OASIS MRI slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

if __package__:
    from .oasis_data import (
        DEFAULT_DATA_SOURCE,
        ROOT,
        MRISliceDataset,
        get_device,
        seed_everything,
    )
else:
    from oasis_data import (  # type: ignore
        DEFAULT_DATA_SOURCE,
        ROOT,
        MRISliceDataset,
        get_device,
        seed_everything,
    )


OUTPUT_DIR = ROOT / "outputs" / "part4_vae"
CHECKPOINT = ROOT / "checkpoints" / "part4_vae.pt"


def encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


def decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ConvolutionalVAE(nn.Module):
    def __init__(self, image_size: int = 128, latent_dim: int = 2):
        super().__init__()
        if image_size < 32 or image_size % 16:
            raise ValueError("image_size must be divisible by 16")
        if latent_dim < 2:
            raise ValueError("latent_dim must be at least 2 for the manifold")

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.feature_size = image_size // 16
        self.encoder = nn.Sequential(
            encoder_block(1, 32),
            encoder_block(32, 64),
            encoder_block(64, 128),
            encoder_block(128, 256),
        )
        features = 256 * self.feature_size**2
        self.fc_mu = nn.Linear(features, latent_dim)
        self.fc_log_var = nn.Linear(features, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, features)
        self.decoder = nn.Sequential(
            decoder_block(256, 128),
            decoder_block(128, 64),
            decoder_block(64, 32),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def encode(self, images: torch.Tensor):
        features = self.encoder(images).flatten(1)
        return self.fc_mu(features), self.fc_log_var(features)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor):
        standard_deviation = torch.exp(0.5 * log_var)
        return mu + standard_deviation * torch.randn_like(standard_deviation)

    def decode(self, latent: torch.Tensor):
        features = self.decoder_input(latent)
        features = features.view(-1, 256, self.feature_size, self.feature_size)
        return self.decoder(features)

    def forward(self, images: torch.Tensor):
        mu, log_var = self.encode(images)
        latent = self.reparameterize(mu, log_var)
        return self.decode(latent), mu, log_var


def vae_loss(
    reconstruction: torch.Tensor,
    images: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
):
    reconstruction_loss = F.binary_cross_entropy_with_logits(
        reconstruction, images, reduction="sum"
    ) / images.size(0)
    kl_divergence = -0.5 * torch.sum(
        1 + log_var - mu.square() - log_var.exp()
    ) / images.size(0)
    return (
        reconstruction_loss + beta * kl_divergence,
        reconstruction_loss,
        kl_divergence,
    )


def make_loaders(args):
    loaders = []
    for split in ("train", "validate", "test"):
        dataset = MRISliceDataset(args.data_source, split, args.image_size)
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


def run_epoch(model, loader, device, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    samples = 0

    for images, _, _ in loader:
        images = images.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            reconstruction, mu, log_var = model(images)
            losses = vae_loss(reconstruction, images, mu, log_var, beta)
            if training:
                losses[0].backward()
                optimizer.step()

        batch_size = images.size(0)
        totals += np.array([loss.item() for loss in losses]) * batch_size
        samples += batch_size

    return tuple(totals / samples)


def save_checkpoint(model, epoch, validation_loss, args):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_loss": float(validation_loss),
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
        },
        args.checkpoint,
    )


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


@torch.inference_mode()
def collect_latent(model, loader, device):
    model.eval()
    latent, slice_indices = [], []
    for images, indices, _ in loader:
        mu, _ = model.encode(images.to(device))
        latent.append(mu.cpu())
        slice_indices.append(indices)
    return torch.cat(latent).numpy(), torch.cat(slice_indices).numpy()


@torch.inference_mode()
def save_visualisations(model, loader, device, output_dir, grid_size=12):
    model.eval()
    images, _, _ = next(iter(loader))
    images = images[:8].to(device)
    reconstruction = torch.sigmoid(model(images)[0])

    figure, axes = plt.subplots(2, len(images), figsize=(16, 4))
    for column in range(len(images)):
        axes[0, column].imshow(images[column, 0].cpu(), cmap="gray")
        axes[1, column].imshow(reconstruction[column, 0].cpu(), cmap="gray")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruction")
    figure.tight_layout()
    figure.savefig(output_dir / "reconstructions.png", dpi=150)
    plt.close(figure)

    latent, slice_indices = collect_latent(model, loader, device)
    figure, axis = plt.subplots(figsize=(7, 6))
    points = axis.scatter(latent[:, 0], latent[:, 1], c=slice_indices, s=10)
    figure.colorbar(points, ax=axis, label="Slice index")
    axis.set(title="OASIS latent manifold", xlabel="z1", ylabel="z2")
    figure.tight_layout()
    figure.savefig(output_dir / "latent_manifold.png", dpi=150)
    plt.close(figure)

    coordinates = torch.linspace(-2.5, 2.5, grid_size, device=device)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    latent_grid = torch.zeros(grid_size**2, model.latent_dim, device=device)
    latent_grid[:, :2] = torch.stack((x.flatten(), y.flatten()), dim=1)
    generated = torch.sigmoid(model.decode(latent_grid)).cpu()[:, 0]
    generated = generated.view(grid_size, grid_size, model.image_size, model.image_size)
    rows = [torch.cat(list(row), dim=1) for row in generated]
    manifold = torch.cat(rows, dim=0)
    plt.imsave(output_dir / "generated_manifold.png", manifold, cmap="gray")


def run(args):
    seed_everything(args.seed)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, test_loader = make_loaders(args)
    model = ConvolutionalVAE(args.image_size, args.latent_dim).to(device)
    history = []

    if args.inference_only:
        checkpoint = load_checkpoint(model, args.checkpoint)
        best_epoch = checkpoint["epoch"]
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_loss = float("inf")
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, device, args.beta, optimizer)
            validation_loss = run_epoch(model, validation_loader, device, args.beta)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss[0],
                    "validation_loss": validation_loss[0],
                }
            )
            if validation_loss[0] < best_loss:
                best_loss = validation_loss[0]
                best_epoch = epoch
                save_checkpoint(model, epoch, best_loss, args)
            print(
                f"Epoch {epoch:02d}/{args.epochs}: "
                f"train={train_loss[0]:.2f}, validation={validation_loss[0]:.2f}"
            )
        load_checkpoint(model, args.checkpoint)

    test_loss = run_epoch(model, test_loader, device, args.beta)
    save_visualisations(model, test_loader, device, args.output_dir)
    metrics = {
        "best_epoch": best_epoch,
        "test_loss": test_loss[0],
        "reconstruction_loss": test_loss[1],
        "kl_divergence": test_loss[2],
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Test loss: {test_loss[0]:.2f}")
    print(f"Visualisations: {args.output_dir}")
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", type=Path, default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference-only", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
