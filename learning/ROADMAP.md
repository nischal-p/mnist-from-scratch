# Roadmap

Status: **current focus — tensor fluency and PyTorch's training mechanics**

## Established foundations

- [x] MNIST examples, train/test splits, PIL images, and labels.
- [x] Pixels as a `28 × 28` NumPy array; flattening to 784 inputs; scaling from
  `0–255` to `0–1`.
- [x] A dense layer as `weights @ inputs + bias`, sigmoid activations, and a
  10-class output vector.
- [x] One-hot targets, mean-squared error, finite-difference gradients, and a
  manual gradient update.
- [x] The broad purpose of backpropagation and outer products for weight
  gradients.
- [x] PyTorch tensors, including `shape`, `dtype`, `device`, and NumPy sharing
  versus copying.
- [x] Image-batch layout: `(batch, channels, height, width)`.
- [x] Matplotlib figures, subplot grids, `imshow`, grayscale images, and
  removing a singleton channel with `squeeze()` for display.

## Practice next

1. Finish the tensor playground exercises: indexing, reshape/view behavior,
   reductions, broadcasting, and matrix multiplication.
2. Verify one actual reshape result by assigning it: `reshaped = x.reshape(...)`.
3. Work through autograd: predict derivatives, explain accumulation in `.grad`,
   and use `optimizer.zero_grad()` correctly.
4. Trace the training-loop rhythm: forward → loss → clear gradients → backward
   → optimizer step → evaluation under `torch.no_grad()`.
5. Train the MNIST PyTorch notebook after fixing its display-cell typo, then
   inspect correct and incorrect predictions.

## Then

1. Build a two-layer MLP for a small synthetic classification task.
2. Understand logits and `CrossEntropyLoss` deeply: shape `(batch, classes)`,
   integer targets `(batch,)`, and no explicit softmax before the loss.
3. Trace shapes through `nn.Conv2d` using MNIST's `(N, 1, 28, 28)` batches.
4. Build a simple CNN for MNIST and compare it with the dense network.
5. Learn embeddings, dot products, softmax, masks, and scaled dot-product
   attention; then implement learned Q/K/V projections.

## Project checkpoints

- **Checkpoint A:** Explain and manually run one training update.
- **Checkpoint B:** Train/debug an MNIST MLP in PyTorch.
- **Checkpoint C:** Build and compare an MNIST CNN.
- **Checkpoint D:** Implement a miniature next-token/attention experiment.
