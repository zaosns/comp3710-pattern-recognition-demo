"""COMP3710 Demo 2 - Part 1: Fourier series and DFT."""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# Names and default values follow the lab sheet.
N = 2048
T = 1.0
f0 = 1
harmonics = [1, 3, 5, 20, 50]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "part1"


def get_device(name="auto"):
    """Select the GPU when one is available."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    return device


def make_time_axis(N, T=1.0, device="cpu"):
    """Create N samples over [0, T), like endpoint=False in the PDF."""
    if N < 1 or T <= 0:
        raise ValueError("N and T must be positive")
    return torch.arange(N, dtype=torch.float32, device=device) * T / N


def square_wave(t):
    """PyTorch version of the square-wave function in the PDF."""
    return torch.sign(torch.sin(2.0 * torch.pi * f0 * t))


def square_wave_fourier(t, f0, N):
    """PyTorch version of the square-wave Fourier series in the PDF."""
    if N < 1:
        raise ValueError("N must be positive")
    result = torch.zeros_like(t)
    for k in range(N):
        n = 2 * k + 1
        result += torch.sin(2 * torch.pi * n * f0 * t) / n
    return (4 / torch.pi) * result


def naive_dft(x):
    """Direct PyTorch DFT on the same device as x; no built-in FFT."""
    if x.ndim != 1 or x.numel() == 0:
        raise ValueError("x must be a non-empty one-dimensional signal")

    N = x.numel()
    x = x.float()
    k = torch.arange(N, dtype=torch.float32, device=x.device).reshape(N, 1)
    n = torch.arange(N, dtype=torch.float32, device=x.device).reshape(1, N)
    angles = -2 * torch.pi * torch.remainder(k * n, N) / N
    real = torch.cos(angles) @ x
    imaginary = torch.sin(angles) @ x
    return torch.complex(real, imaginary)


def naive_dft_gpu(x, device):
    """The explicit GPU version requested in the lab sheet."""
    if device.type not in {"cuda", "mps"}:
        raise ValueError("naive_dft_gpu requires a GPU device")
    return naive_dft(torch.as_tensor(x, dtype=torch.float32, device=device))


def numpy_naive_dft(x):
    """Original direct NumPy DFT retained as a timing baseline."""
    x = np.asarray(x)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty one-dimensional signal")
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X


def synchronize(device):
    """Wait for asynchronous GPU work before reading the timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def time_operation(function, device=None, repeats=3):
    """Return the median runtime and latest result."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    durations = []
    for _ in range(repeats):
        if device is not None:
            synchronize(device)
        start = time.perf_counter()
        result = function()
        if device is not None:
            synchronize(device)
        durations.append(time.perf_counter() - start)
    return float(np.median(durations)), result


def benchmark_dfts(sizes, device, repeats=3):
    """Compare NumPy FFT, NumPy direct DFT, and PyTorch direct DFT."""
    rows = []
    for size in sizes:
        t = make_time_axis(size)
        signal = square_wave_fourier(t, f0, 50).numpy()
        gpu_signal = torch.as_tensor(signal, device=device)

        def torch_dft():
            if device.type in {"cuda", "mps"}:
                return naive_dft_gpu(gpu_signal, device)
            return naive_dft(gpu_signal)

        # Warm up library/GPU kernels before recording short runtimes.
        np.fft.fft(signal)
        torch_dft()
        synchronize(device)

        methods = [
            ("NumPy FFT", "cpu", lambda: np.fft.fft(signal), None),
            ("NumPy direct DFT", "cpu", lambda: numpy_naive_dft(signal), None),
            (
                "PyTorch direct DFT",
                device.type,
                torch_dft,
                device,
            ),
        ]
        results = []
        for method, used_device, function, timing_device in methods:
            seconds, result = time_operation(function, timing_device, repeats)
            if isinstance(result, torch.Tensor):
                result = result.cpu().numpy()
            results.append((method, used_device, seconds, result))

        ranks = {
            item[0]: rank
            for rank, item in enumerate(sorted(results, key=lambda item: item[2]), 1)
        }
        fft_result = results[0][3]
        for method, used_device, seconds, result in results:
            rows.append(
                {
                    "samples": size,
                    "rank": ranks[method],
                    "method": method,
                    "device": used_device,
                    "seconds": seconds,
                    "matches_fft": bool(
                        np.allclose(result, fft_result, rtol=2e-3, atol=2e-2)
                    ),
                }
            )
    return rows


def save_results(t, square, signal, rows):
    """Save the three figures and the timing table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_plot = t.cpu().numpy()
    square_plot = square.cpu().numpy()

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.plot(t_plot, square_plot, "k", label="Square wave")
    plt.title("Original square wave")
    for i, Nh in enumerate(harmonics, start=2):
        y = square_wave_fourier(t, f0, Nh).cpu().numpy()
        plt.subplot(2, 3, i)
        plt.plot(t_plot, y, label=f"N={Nh} harmonics")
        plt.plot(t_plot, square_plot, "k--", alpha=0.5, label="Square wave")
        plt.title(f"Fourier Approximation with N={Nh}")
    for axis in plt.gcf().axes:
        axis.set_ylim(-1.5, 1.5)
        axis.grid(alpha=0.3)
        axis.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reconstructions.png", dpi=180)
    plt.close()

    dft_result = naive_dft(signal).cpu().numpy()
    xf = np.fft.fftfreq(N, d=T / N)[: N // 2]
    magnitude = 2 / N * np.abs(dft_result[: N // 2])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(t_plot, signal.cpu().numpy(), color="c")
    ax1.set_title("Input Square Wave Signal")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, 1.0)
    ax1.grid(True)

    ax2.stem(xf, magnitude, basefmt=" ")
    ax2.set_title("Discrete Fourier Transform (Magnitude Spectrum)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_xlim(0, 50)
    ax2.grid(True)

    # Mark the odd harmonics used to construct the square wave, as in the PDF.
    for i in range(20):
        if i < len(xf) and i % 2 == 1:
            label = f"f{i}: {i}*f0 = {xf[i]:.1f} Hz" if i in {1, 3, 5} else None
            ax2.axvline(
                xf[i],
                color="r",
                linestyle="--",
                alpha=0.7,
                label=label,
            )
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spectrum.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    for method in {row["method"] for row in rows}:
        selected = [row for row in rows if row["method"] == method]
        plt.plot(
            [row["samples"] for row in selected],
            [row["seconds"] for row in selected],
            "o-",
            label=method,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Samples N")
    plt.ylabel("Seconds")
    plt.title("DFT runtime comparison")
    plt.grid(which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "benchmark.png", dpi=180)
    plt.close()

    with (OUTPUT_DIR / "benchmark.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "device": rows[2]["device"],
                "all_results_match_fft": all(row["matches_fft"] for row in rows),
            },
            indent=2,
        )
        + "\n"
    )


def main(device_name="auto", repeats=3, show=False):
    device = get_device(device_name)
    t = make_time_axis(N, T, device)
    square = square_wave(t)
    signal = square_wave_fourier(t, f0, 50)
    rows = benchmark_dfts([256, 512, 1024, 2048], device, repeats)
    save_results(t, square, signal, rows)

    print(f"PyTorch device: {device}")
    print("samples | rank | method               | device | seconds    | matches FFT")
    print("-" * 80)
    for row in sorted(rows, key=lambda item: (item["samples"], item["rank"])):
        print(
            f"{row['samples']:7d} | {row['rank']:4d} | {row['method']:20s} | "
            f"{row['device']:6s} | {row['seconds']:10.6f} | {row['matches_fft']}"
        )

    if show:
        for filename in ("reconstructions.png", "spectrum.png", "benchmark.png"):
            plt.figure(figsize=(12, 7))
            plt.imshow(plt.imread(OUTPUT_DIR / filename))
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.device, args.repeats, args.show)
