# Learner profile

This is a practical snapshot, not a scorecard. It describes durable background,
teaching preferences, and conceptual status. Use `CURRENT_STATE.md` for the
next-session handoff. Update this profile when a misconception is resolved, a
new skill is demonstrated, or a recurring confusion emerges.

## Background

- Experienced software engineer, especially in backend and distributed systems.
- Comfortable with Java, Python, TypeScript, some C++, and production concerns
  such as reliability, performance, event-driven processing, and ownership.
- Newer to machine learning and its mathematical vocabulary.

Teach at an experienced-engineer level: do not re-teach basic programming, but
do define ML and mathematical terms instead of assuming they are familiar.

## Current strengths

- Connects an MNIST image to pixels, a flattened input vector, weights, an
  activation function, outputs, labels, and loss.
- Has built a `784 → 16 → 10` NumPy MNIST network and manually worked through
  a forward pass, one-hot targets, MSE, finite differences, outer products, and
  a gradient update.
- Understands backpropagation's broad chain-rule story, while matrix
  orientations and transposes still need repetition.
- Has discussed sigmoid, ReLU, leaky ReLU, GELU, softmax, and the idea that a
  framework loss can consume logits directly.
- Understands the usual image-tensor convention:
  `(batch, channels, height, width)`.
- Can distinguish grayscale MNIST batches `(N, 1, 28, 28)` from RGB batches
  `(N, 3, 28, 28)`.
- Understands the figure/subplot relationship and why `squeeze()` makes a
  `(1, 28, 28)` grayscale image displayable as `(28, 28)`.
- Is actively asking about code semantics rather than treating examples as
  recipes to copy.

## Work in progress

- Make reshape behavior concrete: calling `reshape` returns a tensor; it does
  not change a variable unless the result is assigned (or an in-place API is
  used).
- Build reliable intuition for which tensor dimensions are batch, channel,
  feature, token, and class dimensions.
- Move from manual gradients to PyTorch autograd without losing the link to the
  underlying chain rule.
- Practice writing and modifying a complete training loop from memory.
- Develop a firm model of logits, cross-entropy, numerical stability, and why
  classification usually uses cross-entropy rather than MSE.

## Teaching preferences inferred so far

- Start with an intuitive explanation, then tie it directly to the exact code.
- Show the shape of each important value; do not skip shape explanations.
- Use small executable experiments, predictions before execution, and shape
  tracing.
- Prefer a clear path toward transformers over broad, disconnected API coverage.
