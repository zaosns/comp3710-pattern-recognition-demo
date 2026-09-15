import numpy as np
import pytest
import torch

from part1_dft.main import (
    benchmark_dfts,
    make_time_axis,
    naive_dft,
    naive_dft_gpu,
    numpy_naive_dft,
    square_wave,
    square_wave_fourier,
)


def test_time_axis_excludes_periodic_endpoint():
    t = make_time_axis(4)
    torch.testing.assert_close(t, torch.tensor([0.0, 0.25, 0.5, 0.75]))


def test_square_wave_has_expected_levels():
    values = square_wave(make_time_axis(128))
    assert set(values.unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_fourier_reconstruction_uses_odd_terms():
    t = make_time_axis(64)
    expected = (4 / torch.pi) * (
        torch.sin(2 * torch.pi * t)
        + torch.sin(2 * torch.pi * 3 * t) / 3
        + torch.sin(2 * torch.pi * 5 * t) / 5
    )
    torch.testing.assert_close(square_wave_fourier(t, 1, 3), expected)


def test_numpy_direct_dft_matches_fft():
    signal = np.random.default_rng(3710).normal(size=32)
    np.testing.assert_allclose(numpy_naive_dft(signal), np.fft.fft(signal), atol=1e-10)


def test_torch_direct_dft_matches_fft_on_cpu():
    signal = torch.randn(32)
    actual = naive_dft(signal).numpy()
    np.testing.assert_allclose(actual, np.fft.fft(signal.numpy()), atol=2e-4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS is unavailable"
)
def test_explicit_gpu_dft_on_mps():
    signal = square_wave_fourier(make_time_axis(256), 1, 20)
    actual = naive_dft_gpu(signal, torch.device("mps")).cpu().numpy()
    np.testing.assert_allclose(actual, np.fft.fft(signal.numpy()), atol=2e-2)


def test_spectrum_contains_odd_harmonics():
    signal = square_wave_fourier(make_time_axis(512), 1, 5)
    spectrum = np.abs(np.fft.fft(signal.numpy()))
    assert all(spectrum[index] > 1 for index in [1, 3, 5, 7, 9])
    assert all(spectrum[index] < 1e-4 for index in [2, 4, 6, 8, 10])


def test_benchmark_methods_match_fft():
    rows = benchmark_dfts([32], torch.device("cpu"), repeats=1)
    assert len(rows) == 3
    assert all(row["matches_fft"] for row in rows)


@pytest.mark.parametrize("function", [naive_dft, numpy_naive_dft])
def test_empty_dft_input_is_rejected(function):
    with pytest.raises(ValueError):
        function(torch.tensor([]) if function is naive_dft else np.array([]))
