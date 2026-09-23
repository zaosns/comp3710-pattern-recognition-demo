"""COMP3710 Demo 2 - Part 2: Eigenfaces."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part2"


def compute_pca(X_train, X_test, n_components):
    """Centre the data, compute the SVD, and project into face space."""
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_test -= mean

    # NumPy returns V transpose as the third SVD output.
    U, S, V = np.linalg.svd(X_train, full_matrices=False)
    components = V[:n_components]

    X_transformed = np.dot(X_train, components.T)
    X_test_transformed = np.dot(X_test, components.T)

    return mean, U, S, V, components, X_transformed, X_test_transformed


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits."""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def main(show=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the Labeled Faces in the Wild dataset as NumPy arrays.
    lfw_people = fetch_lfw_people(
        data_home=DATA_DIR,
        min_faces_per_person=70,
        resize=0.4,
    )

    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    n_features = X.shape[1]
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    n_components = 150
    _, _, S, _, components, X_transformed, X_test_transformed = compute_pca(
        X_train,
        X_test,
        n_components,
    )
    eigenfaces = components.reshape((n_components, h, w))

    print(X_transformed.shape)
    print(X_test_transformed.shape)

    # Plot the first 12 eigenfaces.
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    plt.savefig(OUTPUT_DIR / "eigenfaces.png", dpi=180, bbox_inches="tight")

    # Plot cumulative explained variance (compactness).
    explained_variance = (S**2) / (X_train.shape[0] - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var
    ratio_cumsum = np.cumsum(explained_variance_ratio)
    print(ratio_cumsum.shape)
    eigenvalueCount = np.arange(n_components)

    plt.figure()
    plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
    plt.title("Compactness")
    plt.savefig(OUTPUT_DIR / "compactness.png", dpi=180)

    # Classify the PCA features.
    estimator = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        max_features=150,
        random_state=42,
    )
    estimator.fit(X_transformed, y_train)

    predictions = estimator.predict(X_test_transformed)
    correct = predictions == y_test
    total_test = len(X_test_transformed)
    accuracy = np.sum(correct) / total_test
    report = classification_report(
        y_test,
        predictions,
        target_names=target_names,
    )

    print("Total Testing", total_test)
    print("Predictions", predictions)
    print("Which Correct:", correct)
    print("Total Correct:", np.sum(correct))
    print("Accuracy:", accuracy)
    print(report)

    (OUTPUT_DIR / "classification_report.txt").write_text(
        f"Accuracy: {accuracy}\n\n{report}",
        encoding="utf-8",
    )
    print(f"Results written to {OUTPUT_DIR}")

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.show)
