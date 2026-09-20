import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from part3_cnn.dawnbench_resnet18 import (
    BasicBlock,
    ResNet18,
    load_checkpoint,
    mixup,
    save_checkpoint,
)
from part3_cnn.lfw_cnn import LFWCNN, get_device, train_epoch


def test_lfw_cnn_matches_pdf_architecture():
    model = LFWCNN(num_classes=7)
    convolutions = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]

    assert len(convolutions) == 2
    assert all(layer.kernel_size == (3, 3) for layer in convolutions)
    assert all(layer.out_channels == 32 for layer in convolutions)
    assert model(torch.randn(4, 1, 50, 37)).shape == (4, 7)


def test_lfw_training_updates_weights():
    model = LFWCNN(num_classes=7)
    data = TensorDataset(torch.randn(8, 1, 50, 37), torch.arange(8) % 7)
    loader = DataLoader(data, batch_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.features[0].weight.detach().clone()

    loss, accuracy = train_epoch(
        model, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu")
    )

    assert not torch.equal(before, model.features[0].weight)
    assert loss > 0
    assert 0 <= accuracy <= 1


def test_device_can_be_selected_explicitly():
    assert get_device("cpu") == torch.device("cpu")


def test_resnet18_is_created_from_scratch():
    model = ResNet18()

    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert sum(isinstance(layer, BasicBlock) for layer in model.modules()) == 8
    assert sum(parameter.numel() for parameter in model.parameters()) == 11_173_962


def test_mixup_keeps_batch_shape_and_labels():
    np.random.seed(42)
    X = torch.randn(8, 3, 32, 32)
    y = torch.arange(8)

    mixed_X, first_labels, second_labels, amount = mixup(X, y)

    assert mixed_X.shape == X.shape
    assert torch.equal(first_labels, y)
    assert second_labels.shape == y.shape
    assert 0 <= amount <= 1


def test_checkpoint_round_trip(tmp_path):
    first_model = ResNet18()
    path = tmp_path / "resnet18.pt"
    save_checkpoint(first_model, path, epoch=3, accuracy=0.91)

    second_model = ResNet18()
    checkpoint = load_checkpoint(second_model, path)

    assert checkpoint["epoch"] == 3
    assert checkpoint["validation_accuracy"] == 0.91
    for first, second in zip(first_model.parameters(), second_model.parameters()):
        assert torch.equal(first, second)
