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

## Part 2 — Eigenfaces and Random Forest

### What the code demonstrates

- PCA learns a data-dependent orthonormal basis, unlike the predefined harmonic
  basis used by the Fourier transform.
- Each flattened face is centred with the training mean before NumPy SVD. The
  first 150 rows of `Vh` are the principal directions/eigenfaces.
- Projection `(X - mean) @ components.T` converts each 1,850-pixel face into 150
  face-space features for the Random Forest.
- The split is stratified and the PCA mean/components are fitted only on the
  training split, preventing test-data leakage.

### Likely questions

1. **Why centre the faces?** PCA describes variation around the mean; without
   centring, the first direction can mainly represent overall brightness.
2. **What do U, S, and Vh mean?** For `X = U S Vh`, rows of `Vh` are feature-space
   directions, while squared singular values divided by `n_train - 1` are the
   variances along those directions.
3. **Why use only 150 components?** They provide a compact representation that
   removes low-variance directions while retaining most useful variation.
4. **Why fit PCA before the Random Forest?** PCA supplies compact continuous
   features; the forest then learns nonlinear class decision boundaries.
5. **Why is the test set never used for PCA fitting?** Letting it affect the mean
   or eigenfaces leaks evaluation information and produces an optimistic score.
6. **Why stratify?** It preserves the class proportions in both splits, which is
   important because LFW identity counts are not equal.
7. **Why may eigenfaces have reversed colours compared with another run?** An
   eigenvector and its negative describe the same axis, so SVD component signs
   are not uniquely determined.

### Observed result

- The LFW subset has 1,288 images, 1,850 pixels per image, and seven identities.
- The stratified split contains 966 training and 322 testing images.
- The first 150 components explain 94.65% of training variance.
- Random Forest test accuracy is 64.29%; weighted F1 is 59.67%, while macro F1
  is 46.54%.
- George W Bush has the largest class and 93.98% recall. Several minority
  identities are incorrectly predicted as this majority class, which explains
  why overall accuracy is considerably higher than macro recall/F1.
