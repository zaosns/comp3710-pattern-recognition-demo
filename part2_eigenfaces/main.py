"""COMP3710 Demo 2, Part 2: NumPy eigenfaces and face classification."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part2"


@dataclass(frozen=True)
class PCAResult:
    """Parameters learned from the training split only."""

    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a reproducible stratified train/test split."""
    features = np.asarray(features)
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError("features must have shape [samples, features]")
    if labels.ndim != 1 or labels.size != features.shape[0]:
        raise ValueError("labels must contain one value per sample")
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )


def fit_pca(features: np.ndarray, n_components: int) -> PCAResult:
    """Fit PCA with NumPy SVD without using any test-set information."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must have shape [samples, features]")
    n_samples, n_features = features.shape
    maximum_components = min(n_samples, n_features)
    if n_samples < 2:
        raise ValueError("PCA requires at least two samples")
    if not 1 <= n_components <= maximum_components:
        raise ValueError(f"n_components must be between 1 and {maximum_components}")

    mean = features.mean(axis=0)
    centered = features - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)

    # NumPy returns Vh, whose rows are the orthonormal principal directions.
    components = vh[:n_components]
    explained_variance = singular_values**2 / (n_samples - 1)
    total_variance = explained_variance.sum()
    if total_variance <= 0:
        raise ValueError("PCA is undefined when every sample is identical")
    explained_variance_ratio = explained_variance / total_variance
    return PCAResult(
        mean=mean,
        components=components,
        singular_values=singular_values,
        explained_variance_ratio=explained_variance_ratio,
    )


def transform_pca(features: np.ndarray, pca: PCAResult) -> np.ndarray:
    """Project samples into the PCA space learned from the training split."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != pca.mean.size:
        raise ValueError("features do not match the fitted PCA feature dimension")
    return (features - pca.mean) @ pca.components.T


def plot_eigenfaces(
    pca: PCAResult,
    image_shape: tuple[int, int],
    output_path: Path,
    count: int = 12,
) -> None:
    """Plot the training mean and leading eigenfaces."""
    images = [pca.mean, *pca.components[:count]]
    titles = ["Training mean face"] + [
        f"Eigenface {index}" for index in range(1, len(images))
    ]
    columns = 5
    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3 * rows))
    for axis, image, title in zip(np.ravel(axes), images, titles):
        axis.imshow(image.reshape(image_shape), cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    for axis in np.ravel(axes)[len(images) :]:
        axis.axis("off")
    figure.suptitle("LFW mean face and principal directions")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_compactness(pca: PCAResult, n_components: int, output_path: Path) -> None:
    """Plot cumulative variance explained by the retained eigenfaces."""
    cumulative = np.cumsum(pca.explained_variance_ratio)
    x_values = np.arange(1, n_components + 1)
    components_for_90_percent = int(np.searchsorted(cumulative, 0.9) + 1)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(x_values, cumulative[:n_components])
    axis.axhline(0.9, color="tab:red", linestyle="--", alpha=0.7)
    if components_for_90_percent <= n_components:
        axis.axvline(
            components_for_90_percent,
            color="tab:red",
            linestyle="--",
            alpha=0.7,
            label=f"90% variance: {components_for_90_percent} components",
        )
        axis.legend()
    axis.set_xlabel("Number of principal components")
    axis.set_ylabel("Cumulative explained variance ratio")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("PCA compactness on the LFW training split")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_confusion(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    target_names: np.ndarray,
    output_path: Path,
) -> None:
    """Plot a row-normalised confusion matrix."""
    matrix = confusion_matrix(true_labels, predictions, normalize="true")
    figure, axis = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=target_names)
    display.plot(ax=axis, cmap="Blues", values_format=".2f", colorbar=False)
    axis.set_title("Random Forest on PCA face-space features")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_predictions(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    target_names: np.ndarray,
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["sample", "true_id", "true_name", "predicted_id", "predicted_name"]
        )
        for index, (true_id, predicted_id) in enumerate(zip(true_labels, predictions)):
            writer.writerow(
                [
                    index,
                    int(true_id),
                    target_names[true_id],
                    int(predicted_id),
                    target_names[predicted_id],
                ]
            )


def run_experiment(args: argparse.Namespace) -> None:
    """Download/load LFW, fit PCA and Random Forest, and save evidence."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    lfw = fetch_lfw_people(
        data_home=args.data_dir,
        min_faces_per_person=args.min_faces,
        resize=args.resize,
        download_if_missing=True,
    )
    n_samples, height, width = lfw.images.shape
    x_train, x_test, y_train, y_test = split_dataset(
        lfw.data,
        lfw.target,
        test_size=args.test_size,
        random_state=args.seed,
    )

    pca = fit_pca(x_train, args.components)
    x_train_pca = transform_pca(x_train, pca)
    x_test_pca = transform_pca(x_test, pca)

    classifier = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        max_features=args.components,
        random_state=args.seed,
        n_jobs=-1,
    )
    classifier.fit(x_train_pca, y_train)
    predictions = classifier.predict(x_test_pca)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=lfw.target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        predictions,
        target_names=lfw.target_names,
        zero_division=0,
    )

    plot_eigenfaces(
        pca,
        (height, width),
        args.output_dir / "eigenfaces.png",
    )
    plot_compactness(pca, args.components, args.output_dir / "compactness.png")
    plot_confusion(
        y_test,
        predictions,
        lfw.target_names,
        args.output_dir / "confusion_matrix.png",
    )
    save_predictions(
        y_test,
        predictions,
        lfw.target_names,
        args.output_dir / "predictions.csv",
    )
    (args.output_dir / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    total_counts = np.bincount(lfw.target, minlength=len(lfw.target_names))
    train_counts = np.bincount(y_train, minlength=len(lfw.target_names))
    test_counts = np.bincount(y_test, minlength=len(lfw.target_names))
    class_distribution = {
        name: {
            "total": int(total_counts[index]),
            "train": int(train_counts[index]),
            "test": int(test_counts[index]),
        }
        for index, name in enumerate(lfw.target_names)
    }
    components_for_90_percent = int(
        np.searchsorted(np.cumsum(pca.explained_variance_ratio), 0.9) + 1
    )

    metrics = {
        "dataset": "Labeled Faces in the Wild (funneled)",
        "total_samples": int(n_samples),
        "image_height": int(height),
        "image_width": int(width),
        "features": int(lfw.data.shape[1]),
        "classes": int(len(lfw.target_names)),
        "class_names": lfw.target_names.tolist(),
        "class_distribution": class_distribution,
        "training_samples": int(x_train.shape[0]),
        "testing_samples": int(x_test.shape[0]),
        "stratified_split": True,
        "random_seed": args.seed,
        "pca_components": args.components,
        "cumulative_explained_variance": float(
            np.sum(pca.explained_variance_ratio[: args.components])
        ),
        "components_for_90_percent_variance": components_for_90_percent,
        "random_forest_trees": args.trees,
        "random_forest_max_depth": args.max_depth,
        "accuracy": float(accuracy),
        "classification_report": report,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    print(f"LFW: {n_samples} images, {height}x{width}, {len(lfw.target_names)} classes")
    print(f"Train/test: {len(y_train)}/{len(y_test)} (stratified, seed={args.seed})")
    print(
        f"PCA: {args.components} components explain "
        f"{metrics['cumulative_explained_variance']:.2%} of training variance"
    )
    print(f"Random Forest test accuracy: {accuracy:.2%}")
    print(report_text)
    print(f"Results written to {args.output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-faces", type=int, default=70)
    parser.add_argument("--resize", type=float, default=0.4)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--components", type=int, default=150)
    parser.add_argument("--trees", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
