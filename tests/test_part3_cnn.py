import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from part3_cnn.lfw_cnn import (
    LFWCNN,
    normalise_from_training,
    predict,
    representative_prediction_indices,
    run_epoch,
    select_device,
    stratified_split_indices,
)


def test_cnn_matches_required_architecture_and_output_shape() -> None:
    model = LFWCNN(num_classes=7)
    output = model(torch.randn(4, 1, 50, 37))

    assert output.shape == (4, 7)
    assert (
        len([module for module in model.modules() if isinstance(module, nn.Conv2d)])
        == 2
    )
    assert (
        len(
            [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        )
        == 2
    )
    assert (
        len([module for module in model.modules() if isinstance(module, nn.MaxPool2d)])
        == 2
    )
    assert (
        len([module for module in model.modules() if isinstance(module, nn.Dropout)])
        == 1
    )


def test_cnn_rejects_wrong_image_shape() -> None:
    model = LFWCNN()
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 1, 32, 32))


def test_stratified_splits_are_disjoint_and_cover_every_sample() -> None:
    labels = np.repeat(np.arange(7), 20)
    splits = stratified_split_indices(labels, seed=7)
    split_sets = [set(splits.train), set(splits.validation), set(splits.test)]

    assert split_sets[0].isdisjoint(split_sets[1])
    assert split_sets[0].isdisjoint(split_sets[2])
    assert split_sets[1].isdisjoint(split_sets[2])
    assert set.union(*split_sets) == set(range(len(labels)))
    for indices in (splits.train, splits.validation, splits.test):
        counts = np.bincount(labels[indices], minlength=7)
        assert counts.max() == counts.min()


def test_normalisation_uses_training_statistics_only() -> None:
    images = np.arange(6 * 4 * 3, dtype=np.float32).reshape(6, 4, 3)
    train_indices = np.array([0, 1, 2, 3])
    normalised, mean, standard_deviation = normalise_from_training(
        images, train_indices
    )

    assert np.isclose(normalised[train_indices].mean(), 0.0, atol=1e-6)
    assert np.isclose(normalised[train_indices].std(), 1.0, atol=1e-6)
    changed = images.copy()
    changed[4:] += 10_000
    _, changed_mean, changed_standard_deviation = normalise_from_training(
        changed, train_indices
    )
    assert changed_mean == mean
    assert changed_standard_deviation == standard_deviation


def test_training_epoch_updates_model_parameters() -> None:
    torch.manual_seed(3)
    model = LFWCNN(num_classes=7, hidden_features=16, dropout=0.0)
    loader = DataLoader(
        TensorDataset(torch.randn(8, 1, 50, 37), torch.arange(8) % 7),
        batch_size=4,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.features[0].weight.detach().clone()

    metrics = run_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        optimizer=optimizer,
    )

    assert not torch.equal(before, model.features[0].weight)
    assert metrics["loss"] > 0
    assert 0 <= metrics["accuracy"] <= 1


def test_predict_returns_labels_in_loader_order() -> None:
    model = LFWCNN(num_classes=7, hidden_features=8, dropout=0.0)
    labels = torch.tensor([6, 2, 4])
    loader = DataLoader(TensorDataset(torch.randn(3, 1, 50, 37), labels), batch_size=2)

    actual, predicted = predict(model, loader, torch.device("cpu"))

    np.testing.assert_array_equal(actual, labels.numpy())
    assert predicted.shape == (3,)


def test_representative_predictions_include_successes_and_errors() -> None:
    actual = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    predicted = np.array([0, 1, 1, 0, 2, 3, 3, 2])

    selected = representative_prediction_indices(actual, predicted, 4, count=8)

    assert len(np.unique(selected)) == 8
    assert set(actual[selected]) == {0, 1, 2, 3}
    assert np.any(actual[selected] == predicted[selected])
    assert np.any(actual[selected] != predicted[selected])


def test_device_selection_accepts_cpu() -> None:
    assert select_device("cpu") == torch.device("cpu")
