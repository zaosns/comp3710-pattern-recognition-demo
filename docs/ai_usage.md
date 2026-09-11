# AI usage record

AI assistance is being used as a learning and review aid, as permitted by the
assessment instructions. All generated suggestions are run locally, tested,
reviewed, and must be explainable during the demonstration.

## Part 1

- Used AI to help structure the Fourier/DFT experiment into testable functions.
- Reviewed the assignment's ambiguous use of “harmonics” and documented that
  the parameter represents a count of odd Fourier terms.
- Added independent numerical checks against `numpy.fft.fft` rather than
  accepting the generated direct DFT implementations without verification.
- Added accelerator synchronization so reported GPU timings measure completed
  operations rather than asynchronous dispatch.
- During a second requirements review, identified that the first implementation
  had ported only the DFT to PyTorch. Added and tested the required PyTorch square
  wave and Fourier-series implementations before treating Part 1 as complete.

## Part 2

- Used AI to structure the NumPy PCA and Random Forest experiment into small,
  independently tested functions.
- Corrected two issues in the supplied example: the textual requirement says the
  split is stratified but the shown call omits `stratify=y`, and explained
  variance must use the training sample count rather than the full dataset count.
- Added synthetic tests that ensure held-out data cannot influence the fitted PCA
  mean or principal directions.

## Part 3.1

- Used AI to translate the CNN diagram in the course overview into explicit
  PyTorch modules and to structure the training, validation, and evaluation code.
- Added checks for the specified input shape and architecture, mutually exclusive
  stratified splits, training-only normalisation, gradient updates, and inference
  ordering.
- Kept the model checkpoint selection independent of the test set and retained
  both correct and incorrect predictions for transparent demonstration evidence.
