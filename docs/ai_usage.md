# AI usage record

## Evidence supplied

This record is based on the complete external-AI conversation supplied for the
VAE work.

- **AI service:** ChatGPT
- **Model:** GPT-5.6 Sol
- **Share link:** [ChatGPT conversation](https://chatgpt.com/share/6ab079ee-da70-83ec-9aa7-82cb6bdaa5c0)
- **Local transcript:** [`ai_conversation_vae.txt`](ai_conversation_vae.txt)

The local transcript preserves the supplied conversation text, while the share
link provides the original web-based evidence.

The conversation contains the following actual prompts:

1. **15:39** — “why does the VAE need both reconstruction loss and KL
   divergence?”
2. **15:39 follow-up** — “how should I combine these two losses in PyTorch?”
3. **16:58** — “should I use MSE or binary cross entropy for my VAE
   reconstruction loss?”

## What was learned from the conversation

- Reconstruction loss makes the decoder preserve the content of the input MRI.
- KL divergence regularises each encoded distribution towards a standard normal
  distribution, making the latent space smoother and sampleable.
- The combined objective can be written as
  `reconstruction_loss + beta * kl_divergence`.
- The reparameterised encoder supplies `mu` and `log_var`, and the usual KL term
  uses `1 + log_var - mu**2 - exp(log_var)`.
- Reconstruction-loss choice must be consistent with the input range and the
  decoder output representation.

## How the advice was used and evaluated

The explanation of reconstruction loss, KL divergence and `beta=1.0` was used
to understand and implement the VAE objective. The final implementation reports
the reconstruction and KL terms separately and combines them for
back-propagation.

The external AI recommended MSE as its initial choice for the OASIS MRI
reconstruction. This recommendation was **not copied directly**. After checking
the course VAE example and the actual model interface, the implementation uses
`torch.nn.functional.binary_cross_entropy_with_logits` because:

- input pixels are scaled to `[0, 1]`;
- the decoder returns logits;
- `binary_cross_entropy_with_logits` combines the sigmoid and BCE calculation
  in a numerically stable operation.

The decoder output is passed through `torch.sigmoid` only for visualisation.
The final loss implementation is in
[`part4_recognition/vae.py`](../part4_recognition/vae.py).

## Verification

- The VAE forward pass, loss and gradient flow are covered by the automated
  tests.
- The model was trained on the OASIS data and evaluated on its test split.
- Reconstruction images, a latent scatter plot and a generated two-dimensional
  manifold were inspected after training.
- The complete conversation is retained so the original prompts and AI answers
  can be distinguished from the final implementation decisions.

## Other AI assistance

Codex was also used for repository inspection, PDF/lecture cross-checking,
running tests, identifying stale outputs and updating project documentation.
This summary does not replace any share link or exported prompt history that a
demonstrator may request for that separate interaction.
