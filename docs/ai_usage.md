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
