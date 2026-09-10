"""COMP3710 Demo 2, Part 1: Fourier series and DFT experiments.

The module deliberately implements the DFT formula instead of calling a built-in
FFT. This makes the mathematical operation visible and lets us compare its
quadratic runtime with NumPy's FFT implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import median
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "part1"


def make_time_axis(num_samples: int, duration: float = 1.0) -> np.ndarray:
    """Return evenly spaced samples over a periodic interval [0, duration)."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")
    return np.linspace(0.0, duration, num_samples, endpoint=False)


def square_wave(t: np.ndarray, fundamental_frequency: float = 1.0) -> np.ndarray:
    """Sample a unit-amplitude square wave at times ``t``."""
    if fundamental_frequency <= 0:
        raise ValueError("fundamental_frequency must be positive")
    times = np.asarray(t, dtype=np.float64)
    sine_values = np.sin(2.0 * np.pi * fundamental_frequency * times)
    # A square wave is discontinuous when sine is zero. Use the midpoint value
    # zero there so NumPy float64 and PyTorch float32 follow the same convention.
    return np.where(
        np.isclose(sine_values, 0.0, rtol=0.0, atol=1e-12),
        0.0,
        np.sign(sine_values),
    )


def square_wave_fourier(
    t: np.ndarray,
    fundamental_frequency: float = 1.0,
    num_terms: int = 5,
) -> np.ndarray:
    """Approximate a square wave using the first ``num_terms`` odd harmonics.

    For example, ``num_terms=3`` uses harmonic indices 1, 3, and 5. The
    distinction matters because the assignment's starter code labels the values
    1, 3, and 5 as "harmonics" while using them as counts of Fourier-series terms.
    """
    if fundamental_frequency <= 0:
        raise ValueError("fundamental_frequency must be positive")
    if num_terms < 1:
        raise ValueError("num_terms must be positive")

    times = np.asarray(t, dtype=np.float64)
    odd_indices = 2 * np.arange(num_terms, dtype=np.float64) + 1
    angles = (
        2.0
        * np.pi
        * fundamental_frequency
        * odd_indices[:, np.newaxis]
        * times[np.newaxis, :]
    )
    return (4.0 / np.pi) * np.sum(np.sin(angles) / odd_indices[:, np.newaxis], axis=0)


def torch_time_axis(
    num_samples: int,
    duration: float = 1.0,
    device: str | torch.device = "auto",
) -> torch.Tensor:
    """PyTorch equivalent of :func:`make_time_axis`."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")
    target_device = resolve_device(device) if isinstance(device, str) else device
    return (
        torch.arange(num_samples, dtype=torch.float32, device=target_device)
        * duration
        / num_samples
    )


def _real_tensor(
    values: np.ndarray | torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    target_device = resolve_device(device) if isinstance(device, str) else device
    if isinstance(values, torch.Tensor):
        return values.to(device=target_device, dtype=torch.float32)
    return torch.as_tensor(
        np.asarray(values), dtype=torch.float32, device=target_device
    )


def torch_square_wave(
    t: np.ndarray | torch.Tensor,
    fundamental_frequency: float = 1.0,
    device: str | torch.device = "auto",
) -> torch.Tensor:
    """Generate the square wave using PyTorch operations on ``device``."""
    if fundamental_frequency <= 0:
        raise ValueError("fundamental_frequency must be positive")
    times = _real_tensor(t, device)
    sine_values = torch.sin(2.0 * torch.pi * fundamental_frequency * times)
    return torch.where(
        torch.isclose(
            sine_values,
            torch.zeros_like(sine_values),
            rtol=0.0,
            atol=1e-6,
        ),
        torch.zeros_like(sine_values),
        torch.sign(sine_values),
    )


def torch_square_wave_fourier(
    t: np.ndarray | torch.Tensor,
    fundamental_frequency: float = 1.0,
    num_terms: int = 5,
    device: str | torch.device = "auto",
) -> torch.Tensor:
    """Construct the odd-harmonic Fourier approximation with PyTorch."""
    if fundamental_frequency <= 0:
        raise ValueError("fundamental_frequency must be positive")
    if num_terms < 1:
        raise ValueError("num_terms must be positive")

    times = _real_tensor(t, device)
    odd_indices = (
        2.0 * torch.arange(num_terms, dtype=torch.float32, device=times.device) + 1.0
    )
    angles = (
        2.0 * torch.pi * fundamental_frequency * odd_indices[:, None] * times[None, :]
    )
    return (4.0 / torch.pi) * torch.sum(torch.sin(angles) / odd_indices[:, None], dim=0)


def naive_dft(x: np.ndarray) -> np.ndarray:
    """Compute the one-dimensional DFT directly in O(N^2) time using NumPy."""
    signal = np.asarray(x)
    if signal.ndim != 1:
        raise ValueError("x must be a one-dimensional signal")
    if signal.size == 0:
        raise ValueError("x must not be empty")

    num_samples = signal.size
    sample_indices = np.arange(num_samples)
    result = np.empty(num_samples, dtype=np.complex128)
    for frequency_index in range(num_samples):
        basis = np.exp(-2j * np.pi * frequency_index * sample_indices / num_samples)
        result[frequency_index] = np.sum(signal * basis)
    return result


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit device or select CUDA/MPS/CPU in that order."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator work so benchmark timings are meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def torch_naive_dft(
    x: np.ndarray | torch.Tensor,
    device: str | torch.device = "auto",
) -> torch.Tensor:
    """Compute the DFT using explicit PyTorch tensor operations.

    No PyTorch FFT function is used. The real and imaginary matrix products are
    kept separate so the implementation works on accelerators with limited
    complex-matrix support, including Apple MPS.
    """
    target_device = resolve_device(device) if isinstance(device, str) else device
    if isinstance(x, torch.Tensor):
        target_dtype = torch.complex64 if x.is_complex() else torch.float32
        tensor = x.to(device=target_device, dtype=target_dtype)
    else:
        array = np.asarray(x)
        target_dtype = torch.complex64 if np.iscomplexobj(array) else torch.float32
        tensor = torch.as_tensor(array, dtype=target_dtype, device=target_device)
    if tensor.ndim != 1:
        raise ValueError("x must be a one-dimensional signal")
    if tensor.numel() == 0:
        raise ValueError("x must not be empty")

    if tensor.is_complex():
        real_input = tensor.real.to(torch.float32)
        imaginary_input = tensor.imag.to(torch.float32)
    else:
        real_input = tensor.to(torch.float32)
        imaginary_input = torch.zeros_like(real_input)

    num_samples = tensor.numel()
    indices = torch.arange(num_samples, dtype=torch.float32, device=target_device)
    # exp(-2*pi*i*k*n/N) is periodic in k*n modulo N. Reducing the phase to a
    # single period avoids avoidable float32 trigonometric error for large k*n.
    phase_indices = torch.remainder(indices[:, None] * indices[None, :], num_samples)
    angles = -2.0 * torch.pi * phase_indices / num_samples
    cosine_basis = torch.cos(angles)
    sine_basis = torch.sin(angles)

    real_output = cosine_basis @ real_input - sine_basis @ imaginary_input
    imaginary_output = sine_basis @ real_input + cosine_basis @ imaginary_input
    return torch.complex(real_output, imaginary_output)


def _median_runtime(
    operation: Callable[[], object],
    repeats: int,
    device: torch.device | None = None,
) -> tuple[float, object]:
    if repeats < 1:
        raise ValueError("repeats must be positive")

    durations: list[float] = []
    latest_result: object = None
    for _ in range(repeats):
        if device is not None:
            synchronize_device(device)
        start = time.perf_counter()
        latest_result = operation()
        if device is not None:
            synchronize_device(device)
        durations.append(time.perf_counter() - start)
    return median(durations), latest_result


def benchmark_dfts(
    sizes: Iterable[int],
    num_terms: int = 50,
    repeats: int = 3,
    device: str = "auto",
) -> list[dict[str, object]]:
    """Benchmark direct NumPy DFT, NumPy FFT, and explicit PyTorch DFT."""
    target_device = resolve_device(device)
    rows: list[dict[str, object]] = []

    for num_samples in sizes:
        if num_samples < 1:
            raise ValueError("all benchmark sizes must be positive")
        times = make_time_axis(num_samples)
        signal = square_wave_fourier(times, num_terms=num_terms)
        torch_times = torch_time_axis(num_samples, device=target_device)
        torch_signal = torch_square_wave_fourier(
            torch_times, num_terms=num_terms, device=target_device
        )

        numpy_seconds, numpy_result = _median_runtime(
            lambda: naive_dft(signal), repeats
        )
        fft_seconds, fft_result = _median_runtime(lambda: np.fft.fft(signal), repeats)
        torch_seconds, torch_result = _median_runtime(
            lambda: torch_naive_dft(torch_signal, target_device),
            repeats,
            target_device,
        )

        numpy_array = np.asarray(numpy_result)
        fft_array = np.asarray(fft_result)
        torch_array = torch_result.detach().cpu().numpy()  # type: ignore[union-attr]

        size_rows = [
            {
                "samples": num_samples,
                "method": "NumPy direct DFT",
                "device": "cpu",
                "seconds": numpy_seconds,
                "matches_numpy_fft": bool(
                    np.allclose(numpy_array, fft_array, rtol=1e-7, atol=1e-7)
                ),
            },
            {
                "samples": num_samples,
                "method": "NumPy FFT",
                "device": "cpu",
                "seconds": fft_seconds,
                "matches_numpy_fft": True,
            },
            {
                "samples": num_samples,
                "method": "PyTorch direct DFT",
                "device": target_device.type,
                "seconds": torch_seconds,
                "matches_numpy_fft": bool(
                    np.allclose(torch_array, fft_array, rtol=2e-3, atol=2e-2)
                ),
            },
        ]
        for rank, row in enumerate(
            sorted(size_rows, key=lambda item: float(item["seconds"])), start=1
        ):
            row["speed_rank"] = rank
        rows.extend(size_rows)
    return rows


def plot_reconstructions(
    t: np.ndarray,
    term_counts: Iterable[int],
    output_path: Path,
) -> None:
    """Plot the target square wave and several Fourier approximations."""
    counts = list(term_counts)
    figures = len(counts) + 1
    columns = 3
    rows = int(np.ceil(figures / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4 * rows), squeeze=False)
    flat_axes = axes.ravel()
    target = square_wave(t)

    flat_axes[0].plot(t, target, color="black", label="Target square wave")
    flat_axes[0].set_title("Original square wave")
    for axis, count in zip(flat_axes[1:], counts):
        approximation = square_wave_fourier(t, num_terms=count)
        axis.plot(t, target, "k--", alpha=0.45, label="Target")
        axis.plot(t, approximation, label=f"First {count} odd terms")
        axis.set_title(f"Fourier approximation: {count} terms")

    for axis in flat_axes[:figures]:
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Amplitude")
        axis.set_ylim(-1.5, 1.5)
        axis.grid(True, alpha=0.3)
        axis.legend()
    for axis in flat_axes[figures:]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum(
    signal: np.ndarray,
    sample_spacing: float,
    output_path: Path,
    max_frequency: float = 110.0,
) -> None:
    """Plot the one-sided magnitude spectrum of a real signal."""
    spectrum = naive_dft(signal)
    frequencies = np.fft.rfftfreq(signal.size, d=sample_spacing)
    magnitudes = 2.0 * np.abs(spectrum[: frequencies.size]) / signal.size
    magnitudes[0] /= 2.0
    if signal.size % 2 == 0:
        magnitudes[-1] /= 2.0  # The Nyquist bin has no separate negative pair.

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.stem(frequencies, magnitudes, basefmt=" ")
    axis.set_xlim(0.0, max_frequency)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Amplitude")
    axis.set_title("DFT magnitude spectrum of the Fourier square-wave approximation")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compare_harmonics(
    signal: np.ndarray,
    sample_spacing: float,
    fundamental_frequency: float,
    num_terms: int,
) -> list[dict[str, float | int]]:
    """Compare expected Fourier-series amplitudes with measured DFT bins."""
    spectrum = naive_dft(signal)
    frequency_resolution = 1.0 / (signal.size * sample_spacing)
    nyquist_frequency = 0.5 / sample_spacing
    rows: list[dict[str, float | int]] = []

    for term in range(num_terms):
        harmonic_index = 2 * term + 1
        expected_frequency = harmonic_index * fundamental_frequency
        # A sampled sine exactly at Nyquist is identically zero, and components
        # above Nyquist alias, so neither can be compared with the source term.
        if expected_frequency >= nyquist_frequency:
            break
        bin_index = int(round(expected_frequency / frequency_resolution))
        measured_frequency = bin_index * frequency_resolution
        expected_amplitude = 4.0 / (np.pi * harmonic_index)
        measured_amplitude = 2.0 * np.abs(spectrum[bin_index]) / signal.size
        rows.append(
            {
                "harmonic_index": harmonic_index,
                "expected_frequency_hz": expected_frequency,
                "dft_bin_frequency_hz": measured_frequency,
                "expected_amplitude": expected_amplitude,
                "measured_amplitude": float(measured_amplitude),
                "absolute_error": float(abs(measured_amplitude - expected_amplitude)),
            }
        )
    return rows


def plot_benchmarks(rows: list[dict[str, object]], output_path: Path) -> None:
    """Plot runtime growth for each DFT implementation."""
    fig, axis = plt.subplots(figsize=(9, 6))
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        axis.plot(
            [int(row["samples"]) for row in method_rows],
            [float(row["seconds"]) for row in method_rows],
            marker="o",
            label=method,
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Number of samples N")
    axis.set_ylabel("Median runtime (seconds)")
    axis.set_title("DFT runtime comparison")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_benchmark_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_rows_csv(
    rows: list[dict[str, object]] | list[dict[str, float | int]],
    output_path: Path,
) -> None:
    if not rows:
        raise ValueError("cannot save an empty result table")
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t = make_time_axis(args.samples, args.duration)
    signal = square_wave_fourier(t, args.frequency, num_terms=args.spectrum_terms)
    target_device = resolve_device(args.device)
    torch_t = torch_time_axis(args.samples, args.duration, target_device)
    torch_target = torch_square_wave(torch_t, args.frequency, target_device)
    torch_signal = torch_square_wave_fourier(
        torch_t, args.frequency, args.spectrum_terms, target_device
    )
    synchronize_device(target_device)

    plot_reconstructions(t, args.terms, output_dir / "reconstructions.png")
    plot_spectrum(
        signal,
        sample_spacing=args.duration / args.samples,
        output_path=output_dir / "spectrum.png",
        max_frequency=args.max_frequency,
    )
    harmonic_rows = compare_harmonics(
        signal,
        sample_spacing=args.duration / args.samples,
        fundamental_frequency=args.frequency,
        num_terms=args.spectrum_terms,
    )
    save_rows_csv(harmonic_rows, output_dir / "harmonics.csv")

    benchmark_rows = benchmark_dfts(
        args.benchmark_sizes,
        num_terms=args.spectrum_terms,
        repeats=args.repeats,
        device=args.device,
    )
    save_benchmark_csv(benchmark_rows, output_dir / "benchmark.csv")
    plot_benchmarks(benchmark_rows, output_dir / "benchmark.png")

    summary = {
        "fundamental_frequency_hz": args.frequency,
        "duration_seconds": args.duration,
        "spectrum_samples": args.samples,
        "spectrum_fourier_terms": args.spectrum_terms,
        "highest_harmonic_index": 2 * args.spectrum_terms - 1,
        "benchmark_repeats": args.repeats,
        "selected_torch_device": target_device.type,
        "torch_square_wave_matches_numpy": bool(
            np.allclose(torch_target.cpu().numpy(), square_wave(t, args.frequency))
        ),
        "torch_fourier_series_matches_numpy": bool(
            np.allclose(torch_signal.cpu().numpy(), signal, rtol=2e-4, atol=2e-4)
        ),
        "maximum_harmonic_amplitude_error": max(
            row["absolute_error"] for row in harmonic_rows
        ),
        "all_results_match_numpy_fft": all(
            bool(row["matches_numpy_fft"]) for row in benchmark_rows
        ),
        "fastest_method_by_size": {
            str(size): min(
                (row for row in benchmark_rows if row["samples"] == size),
                key=lambda row: float(row["seconds"]),
            )["method"]
            for size in args.benchmark_sizes
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Results written to {output_dir}")
    print("samples | rank | method               | device | seconds    | matches FFT")
    print("-" * 80)
    for row in sorted(
        benchmark_rows,
        key=lambda item: (int(item["samples"]), int(item["speed_rank"])),
    ):
        print(
            f"{int(row['samples']):7d} | "
            f"{int(row['speed_rank']):4d} | "
            f"{str(row['method']):20s} | "
            f"{str(row['device']):6s} | "
            f"{float(row['seconds']):10.6f} | "
            f"{row['matches_numpy_fft']}"
        )

    if args.show:
        for filename in ("reconstructions.png", "spectrum.png", "benchmark.png"):
            image = plt.imread(output_dir / filename)
            plt.figure(figsize=(12, 7))
            plt.imshow(image)
            plt.axis("off")
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--frequency", type=float, default=1.0)
    parser.add_argument("--terms", type=int, nargs="+", default=[1, 3, 5, 20, 50])
    parser.add_argument("--spectrum-terms", type=int, default=50)
    parser.add_argument(
        "--benchmark-sizes", type=int, nargs="+", default=[256, 512, 1024, 2048]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--max-frequency", type=float, default=110.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    return parser


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
