"""Train a WGAN-GP to generate OASIS brain MRI slices."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

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


OUTPUT_DIR = ROOT / "outputs" / "part4_gan"
CHECKPOINT = ROOT / "checkpoints" / "part4_gan.pt"


class GANMRIDataset(Dataset):
    def __init__(self, source: Path, image_size: int):
        self.slices = MRISliceDataset(source, "train", image_size)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        image, _, _ = self.slices[index]
        return image * 2 - 1


def initialise_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        image_size: int = 128,
        base_channels: int = 32,
    ):
        super().__init__()
        if image_size < 32 or image_size & (image_size - 1):
            raise ValueError("image_size must be a power of two and at least 32")
        self.latent_dim = latent_dim
        self.image_size = image_size
        stages = int(math.log2(image_size)) - 2
        self.initial_channels = base_channels * 2 ** (stages - 1)
        self.project = nn.Linear(latent_dim, self.initial_channels * 4 * 4)

        layers = []
        in_channels = self.initial_channels
        for multiplier in reversed([2**index for index in range(stages - 1)]):
            out_channels = base_channels * multiplier
            layers += [
                nn.ConvTranspose2d(
                    in_channels, out_channels, 4, 2, 1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
        layers += [nn.ConvTranspose2d(in_channels, 1, 4, 2, 1), nn.Tanh()]
        self.network = nn.Sequential(*layers)
        self.apply(initialise_weights)

    def forward(self, latent):
        features = self.project(latent)
        features = features.view(-1, self.initial_channels, 4, 4)
        return self.network(features)


class Critic(nn.Module):
    def __init__(self, image_size: int = 128, base_channels: int = 32):
        super().__init__()
        if image_size < 32 or image_size & (image_size - 1):
            raise ValueError("image_size must be a power of two and at least 32")
        self.image_size = image_size
        stages = int(math.log2(image_size)) - 2
        layers = []
        in_channels = 1
        for stage in range(stages):
            out_channels = base_channels * 2**stage
            layers += [
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.score = nn.Linear(in_channels * 4 * 4, 1)
        self.apply(initialise_weights)

    def forward(self, images):
        return self.score(self.features(images).flatten(1)).flatten()


def gradient_penalty(critic, real_images, fake_images):
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_images.device)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)
    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        scores,
        interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
    )[0]
    gradients = gradients.flatten(1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
def update_ema(ema_generator, generator, decay):
    for ema, current in zip(ema_generator.parameters(), generator.parameters()):
        ema.mul_(decay).add_(current, alpha=1 - decay)
    for ema, current in zip(ema_generator.buffers(), generator.buffers()):
        ema.copy_(current)


def make_loader(args):
    dataset = GANMRIDataset(args.data_source, args.image_size)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    generator,
    critic,
    ema_generator,
    loader,
    generator_optimizer,
    critic_optimizer,
    device,
    args,
):
    generator.train()
    critic.train()
    critic_total = generator_total = 0.0
    critic_updates = generator_updates = 0

    for batch_index, real_images in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        critic_optimizer.zero_grad(set_to_none=True)
        latent = torch.randn(batch_size, args.latent_dim, device=device)
        with torch.no_grad():
            fake_images = generator(latent)
        penalty = gradient_penalty(critic, real_images, fake_images)
        critic_loss = (
            critic(fake_images).mean()
            - critic(real_images).mean()
            + args.gradient_penalty * penalty
        )
        critic_loss.backward()
        critic_optimizer.step()
        critic_total += critic_loss.item()
        critic_updates += 1

        final_smoke_batch = args.max_batches and batch_index + 1 == args.max_batches
        if (batch_index + 1) % args.critic_steps == 0 or final_smoke_batch:
            generator_optimizer.zero_grad(set_to_none=True)
            latent = torch.randn(batch_size, args.latent_dim, device=device)
            generator_loss = -critic(generator(latent)).mean()
            generator_loss.backward()
            generator_optimizer.step()
            update_ema(ema_generator, generator, args.ema_decay)
            generator_total += generator_loss.item()
            generator_updates += 1

    return critic_total / critic_updates, generator_total / generator_updates


def save_checkpoint(generator, critic, ema_generator, fixed_noise, epoch, args):
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "ema_generator_state_dict": ema_generator.state_dict(),
            "fixed_noise": fixed_noise.cpu(),
            "epoch": epoch,
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
        },
        args.checkpoint,
    )


def load_checkpoint(generator, critic, ema_generator, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    ema_generator.load_state_dict(
        checkpoint.get("ema_generator_state_dict", checkpoint["generator_state_dict"])
    )
    return checkpoint


@torch.inference_mode()
def generate_images(generator, latent):
    generator.eval()
    return ((generator(latent) + 1) / 2).clamp(0, 1)


def save_loss_plot(history, path):
    epochs = [row["epoch"] for row in history]
    plt.plot(epochs, [row["critic_loss"] for row in history], label="Critic")
    plt.plot(epochs, [row["generator_loss"] for row in history], label="Generator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(args):
    seed_everything(args.seed)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    loader = make_loader(args)
    generator = Generator(
        args.latent_dim, args.image_size, args.base_channels
    ).to(device)
    critic = Critic(args.image_size, args.base_channels).to(device)
    ema_generator = copy.deepcopy(generator).eval().requires_grad_(False)
    fixed_noise = torch.randn(args.sample_count, args.latent_dim, device=device)
    history = []

    if args.inference_only:
        checkpoint = load_checkpoint(
            generator, critic, ema_generator, args.checkpoint
        )
        fixed_noise = checkpoint["fixed_noise"].to(device)
        completed_epochs = checkpoint["epoch"]
    else:
        generator_optimizer = torch.optim.Adam(
            generator.parameters(), lr=args.learning_rate, betas=(0.0, 0.9)
        )
        critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=args.learning_rate, betas=(0.0, 0.9)
        )
        completed_epochs = 0
        for epoch in range(1, args.epochs + 1):
            critic_loss, generator_loss = train_one_epoch(
                generator,
                critic,
                ema_generator,
                loader,
                generator_optimizer,
                critic_optimizer,
                device,
                args,
            )
            completed_epochs = epoch
            history.append(
                {
                    "epoch": epoch,
                    "critic_loss": critic_loss,
                    "generator_loss": generator_loss,
                }
            )
            save_checkpoint(
                generator, critic, ema_generator, fixed_noise, epoch, args
            )
            if epoch == 1 or epoch % args.sample_every == 0:
                images = generate_images(ema_generator, fixed_noise)
                save_image(
                    images,
                    args.output_dir / f"samples_epoch_{epoch:03d}.png",
                    nrow=8,
                )
            print(
                f"Epoch {epoch:03d}/{args.epochs}: "
                f"critic={critic_loss:.4f}, generator={generator_loss:.4f}"
            )

    images = generate_images(ema_generator, fixed_noise)
    save_image(images, args.output_dir / "generated_samples.png", nrow=8)
    if history:
        save_loss_plot(history, args.output_dir / "training_losses.png")

    # This simple score and the sample grid help check for mode collapse.
    diversity = images.std(dim=0).mean().item()
    metrics = {
        "completed_epochs": completed_epochs,
        "mean_pixel_standard_deviation": diversity,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Diversity score: {diversity:.4f}")
    print(f"Generated samples: {args.output_dir / 'generated_samples.png'}")
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", type=Path, default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-penalty", type=float, default=10.0)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference-only", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
