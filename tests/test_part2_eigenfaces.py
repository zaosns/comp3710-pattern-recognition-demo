import numpy as np
import pytest

from part2_eigenfaces.main import fit_pca, split_dataset, transform_pca


def test_stratified_split_preserves_every_class() -> None:
    features = np.arange(240, dtype=float).reshape(60, 4)
    labels = np.repeat([0, 1, 2], 20)
    _, _, y_train, y_test = split_dataset(
        features, labels, test_size=0.25, random_state=42
    )
    assert np.bincount(y_train).tolist() == [15, 15, 15]
    assert np.bincount(y_test).tolist() == [5, 5, 5]


def test_pca_uses_training_mean_and_orthonormal_components() -> None:
    training = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 1.0, 0.0], [6.0, 1.0, 0.0]]
    )
    pca = fit_pca(training, n_components=2)
    np.testing.assert_allclose(pca.mean, training.mean(axis=0))
    np.testing.assert_allclose(pca.components @ pca.components.T, np.eye(2), atol=1e-12)
    assert np.all(np.diff(pca.singular_values) <= 0)


def test_pca_does_not_modify_training_array() -> None:
    rng = np.random.default_rng(3710)
    training = rng.normal(size=(12, 5))
    unchanged = training.copy()
    fit_pca(training, n_components=3)
    np.testing.assert_array_equal(training, unchanged)


def test_test_data_cannot_influence_fitted_pca() -> None:
    training = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    extreme_test_sample = np.array([[1000.0, -1000.0]])
    pca = fit_pca(training, n_components=2)
    np.testing.assert_allclose(pca.mean, [1.0, 1.0 / 3.0])
    projected = transform_pca(extreme_test_sample, pca)
    assert projected.shape == (1, 2)
    np.testing.assert_allclose(pca.mean, [1.0, 1.0 / 3.0])


def test_explained_variance_ratio_is_valid() -> None:
    rng = np.random.default_rng(3710)
    pca = fit_pca(rng.normal(size=(20, 8)), n_components=4)
    assert np.all(pca.explained_variance_ratio >= 0)
    np.testing.assert_allclose(pca.explained_variance_ratio.sum(), 1.0)
    cumulative = np.cumsum(pca.explained_variance_ratio)
    assert np.all(np.diff(cumulative) >= 0)


def test_transform_matches_manual_center_and_projection() -> None:
    training = np.array([[0.0, 1.0], [2.0, 1.0], [4.0, 3.0]])
    pca = fit_pca(training, n_components=2)
    expected = (training - pca.mean) @ pca.components.T
    np.testing.assert_allclose(transform_pca(training, pca), expected)


@pytest.mark.parametrize("components", [0, 4])
def test_invalid_component_count_is_rejected(components: int) -> None:
    with pytest.raises(ValueError):
        fit_pca(np.ones((3, 5)) + np.arange(3)[:, None], components)


def test_identical_samples_are_rejected() -> None:
    with pytest.raises(ValueError):
        fit_pca(np.ones((5, 3)), n_components=2)
