import torch
from torch.nn import functional as F

from part4_recognition.gan import Critic, Generator, gradient_penalty
from part4_recognition.unet import (
    NUM_CLASSES,
    UNet,
    hard_dice_totals,
    segmentation_loss,
)
from part4_recognition.vae import ConvolutionalVAE, vae_loss


def test_vae_forward_and_elbo_support_backpropagation() -> None:
    model = ConvolutionalVAE(image_size=32, latent_dim=2)
    images = torch.rand(2, 1, 32, 32)

    reconstruction, mu, log_var = model(images)
    loss, reconstruction_loss, kl_divergence = vae_loss(
        reconstruction, images, mu, log_var
    )
    loss.backward()

    assert reconstruction.shape == images.shape
    assert mu.shape == log_var.shape == (2, 2)
    assert torch.isfinite(loss)
    assert reconstruction_loss > 0
    assert kl_divergence >= 0
    assert model.fc_mu.weight.grad is not None


def test_unet_outputs_four_categorical_channels_and_finite_loss() -> None:
    model = UNet(num_classes=NUM_CLASSES, base_channels=8)
    images = torch.randn(2, 1, 32, 32)
    class_indices = torch.randint(0, NUM_CLASSES, (2, 32, 32))
    one_hot_targets = F.one_hot(class_indices, NUM_CLASSES).permute(0, 3, 1, 2)
    one_hot_targets = one_hot_targets.to(torch.float32)

    logits = model(images)
    loss = segmentation_loss(logits, one_hot_targets, dice_weight=0.5)
    loss.backward()

    assert logits.shape == (2, NUM_CLASSES, 32, 32)
    assert torch.all(one_hot_targets.sum(dim=1) == 1)
    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None


def test_hard_dice_is_one_for_perfect_segmentation() -> None:
    class_indices = torch.randint(0, NUM_CLASSES, (2, 16, 16))
    one_hot = F.one_hot(class_indices, NUM_CLASSES).permute(0, 3, 1, 2)
    one_hot = one_hot.to(torch.float32)
    logits = one_hot.mul(20.0).sub(10.0)

    intersection, denominator = hard_dice_totals(logits, one_hot)
    dice = 2.0 * intersection / denominator

    torch.testing.assert_close(dice, torch.ones(NUM_CLASSES))


def test_gan_generator_critic_and_gradient_penalty_support_backpropagation() -> None:
    generator = Generator(latent_dim=8, image_size=32, base_channels=8)
    critic = Critic(image_size=32, base_channels=8)
    latent = torch.randn(2, 8)
    real_images = torch.randn(2, 1, 32, 32).clamp(-1.0, 1.0)

    fake_images = generator(latent)
    fake_scores = critic(fake_images)
    penalty = gradient_penalty(critic, real_images, fake_images.detach())
    generator_loss = -fake_scores.mean()
    generator_loss.backward()

    assert fake_images.shape == real_images.shape
    assert fake_scores.shape == (2,)
    assert torch.isfinite(penalty)
    assert penalty >= 0
    assert generator.project.weight.grad is not None
