# Demo notes

## Part 1 — Fourier series and DFT

### What the code demonstrates

- A square wave can be represented as a sum of odd sine harmonics:
  `4/pi * sum(sin(2*pi*(2k+1)*f0*t)/(2k+1))`.
- At its discontinuities the ideal square wave has no unique value; this code
  consistently uses the midpoint value zero for both NumPy and PyTorch.
- Increasing the number of terms improves the flat regions and sharpens the
  transition, but the overshoot near each discontinuity remains because of the
  Gibbs phenomenon.
- The direct DFT projects the signal onto complex exponential basis vectors and
  requires `N * N` operations, giving `O(N^2)` complexity.
- FFT exploits symmetry and repeated subproblems, reducing complexity to
  `O(N log N)`.
- The explicit PyTorch DFT can run its matrix operations in parallel on a GPU,
  but device dispatch and memory overhead can make it slower for small signals.

### Likely questions

1. **Why are only odd peaks visible?** A symmetric 50% duty-cycle square wave
   has zero coefficients for even harmonics.
2. **Why do peak amplitudes decrease?** The amplitude of harmonic `n` is
   proportional to `1/n`.
3. **Why does the 50-term approximation still overshoot?** Fourier partial sums
   exhibit Gibbs ringing at discontinuities; more terms narrow the ringing but
   do not remove the limiting overshoot.
4. **Why is FFT fastest?** Its algorithmic complexity is lower than evaluating
   every input-frequency pair directly.
5. **Why synchronize the GPU while timing?** Accelerator operations are queued
   asynchronously, so timing without synchronization measures dispatch rather
   than completed computation.
6. **What does `num_terms=50` mean here?** It uses 50 odd terms with indices
   `1, 3, ..., 99`; it does not mean harmonic index 50.
7. **Why might measured and expected frequency components differ?** Exact
   agreement is expected here because the frequencies align with DFT bins. In a
   general sampled signal, finite precision, spectral leakage from non-bin-
   aligned frequencies, aliasing above Nyquist, and windowing can create
   differences. A real signal also has conjugate components at positive and
   negative frequencies, so the one-sided plot doubles non-DC magnitudes.
