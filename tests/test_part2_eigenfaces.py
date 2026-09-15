import numpy as np

from part2_eigenfaces.main import compute_pca


def test_pca_shapes_follow_the_pdf_pipeline():
    rng = np.random.default_rng(3710)
    X_train = rng.normal(size=(12, 6))
    X_test = rng.normal(size=(4, 6))

    mean, U, S, V, components, X_transformed, X_test_transformed = compute_pca(
        X_train,
        X_test,
        n_components=3,
    )

    assert mean.shape == (6,)
    assert U.shape == (12, 6)
    assert S.shape == (6,)
    assert V.shape == (6, 6)
    assert components.shape == (3, 6)
    assert X_transformed.shape == (12, 3)
    assert X_test_transformed.shape == (4, 3)


def test_pca_projection_uses_the_training_mean():
    X_train = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 2.0]])
    X_test = np.array([[8.0, 3.0]])
    original_X_train = X_train.copy()
    original_X_test = X_test.copy()

    mean, _, _, _, components, X_transformed, X_test_transformed = compute_pca(
        X_train,
        X_test,
        n_components=2,
    )

    np.testing.assert_allclose(mean, original_X_train.mean(axis=0))
    np.testing.assert_allclose(X_train, original_X_train - mean)
    np.testing.assert_allclose(X_test, original_X_test - mean)
    np.testing.assert_allclose(X_transformed, X_train @ components.T)
    np.testing.assert_allclose(X_test_transformed, X_test @ components.T)
