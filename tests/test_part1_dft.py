import numpy as np
import pytest
import torch

from part1_dft.main import (
    compare_harmonics,
    make_time_axis,
    naive_dft,
    square_wave,
    square_wave_fourier,
    torch_naive_dft,
    torch_square_wave,
    torch_square_wave_fourier,
    torch_time_axis,
)


def test_time_axis_excludes_periodic_endpoint() -> None:
    times = make_time_axis(4, duration=1.0)
    np.testing.assert_allclose(times, [0.0, 0.25, 0.5, 0.75])


def test_square_wave_has_expected_levels() -> None:
    values = square_wave(make_time_axis(128))
    assert set(np.unique(values)).issubset({-1.0, 0.0, 1.0})


def test_fourier_reconstruction_uses_first_odd_terms() -> None:
    times = make_time_axis(64)
    expected = (4.0 / np.pi) * (
        np.sin(2.0 * np.pi * times)
        + np.sin(2.0 * np.pi * 3.0 * times) / 3.0
        + np.sin(2.0 * np.pi * 5.0 * times) / 5.0
    )
    np.testing.assert_allclose(square_wave_fourier(times, num_terms=3), expected)


def test_pytorch_wave_functions_match_numpy_on_cpu() -> None:
    numpy_times = make_time_axis(512)
    torch_times = torch_time_axis(512, device="cpu")
    np.testing.assert_allclose(torch_times.numpy(), numpy_times, atol=1e-7)
    np.testing.assert_allclose(
        torch_square_wave(torch_times, device="cpu").numpy(),
        square_wave(numpy_times),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        torch_square_wave_fourier(torch_times, num_terms=50, device="cpu").numpy(),
        square_wave_fourier(numpy_times, num_terms=50),
        rtol=2e-4,
        atol=2e-4,
    )


def test_direct_numpy_dft_matches_fft() -> None:
    rng = np.random.default_rng(3710)
    signal = rng.normal(size=32)
    np.testing.assert_allclose(naive_dft(signal), np.fft.fft(signal), atol=1e-10)


def test_explicit_torch_dft_matches_fft_on_cpu() -> None:
    rng = np.random.default_rng(3710)
    signal = rng.normal(size=32) + 1j * rng.normal(size=32)
    actual = torch_naive_dft(signal, device="cpu").numpy()
    np.testing.assert_allclose(actual, np.fft.fft(signal), rtol=1e-4, atol=2e-4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS is unavailable"
)
def test_explicit_torch_dft_matches_fft_on_mps() -> None:
    signal = square_wave_fourier(make_time_axis(1024), num_terms=50)
    actual = torch_naive_dft(signal, device="mps").cpu().numpy()
    np.testing.assert_allclose(actual, np.fft.fft(signal), rtol=2e-3, atol=2e-2)


def test_fourier_square_wave_spectrum_contains_odd_harmonics() -> None:
    times = make_time_axis(512)
    spectrum = np.abs(np.fft.fft(square_wave_fourier(times, num_terms=5)))
    assert all(spectrum[index] > 1.0 for index in [1, 3, 5, 7, 9])
    assert all(spectrum[index] < 1e-8 for index in [2, 4, 6, 8, 10])


def test_measured_harmonics_match_fourier_series_coefficients() -> None:
    times = make_time_axis(512)
    signal = square_wave_fourier(times, num_terms=5)
    rows = compare_harmonics(signal, 1.0 / 512, 1.0, num_terms=5)
    assert [row["harmonic_index"] for row in rows] == [1, 3, 5, 7, 9]
    assert max(row["absolute_error"] for row in rows) < 1e-12


@pytest.mark.parametrize("num_samples", [0, -1])
def test_invalid_sample_count_is_rejected(num_samples: int) -> None:
    with pytest.raises(ValueError):
        make_time_axis(num_samples)


def test_non_positive_fourier_terms_are_rejected() -> None:
    with pytest.raises(ValueError):
        square_wave_fourier(make_time_axis(8), num_terms=0)


@pytest.mark.parametrize("implementation", [naive_dft, torch_naive_dft])
def test_empty_dft_input_is_rejected(implementation) -> None:
    with pytest.raises(ValueError):
        implementation(np.array([]))
