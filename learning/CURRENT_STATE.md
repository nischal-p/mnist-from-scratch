# Current state

Last substantial update: **2026-07-21**

## Where we are

The active learning artifact is `notebooks/02-pytorch-playground.ipynb`.
Current focus: tensor fluency—especially image batch shapes, plotting, reshape
behavior, broadcasting, reductions, and matrix multiplication.

Recent confirmed understanding:

- `(4, 1, 28, 28)` means 4 grayscale images; `(4, 3, 28, 28)` means 4 RGB
  images.
- A Matplotlib figure is the overall canvas; subplots are plotting areas in its
  grid.
- `image_batch[i].squeeze()` changes a single image from `(1, 28, 28)` to
  `(28, 28)` for grayscale display.

## Best next session

Continue the playground's **Indexing, reshaping, and views** section.

1. Predict the result of each `flatten(start_dim=...)` call before running it.
2. Fix the exercise by assigning `my_var = my_var.reshape(3, 8)`, then explain
   why `(3, 7)` cannot contain the 24 elements.
3. Complete the row-wise squared-length exercise and state its input/output
   shapes.
4. Proceed to autograd only after these shape operations feel predictable.

## Known code notes (do not fix automatically)

- `notebooks/03-mnist-using-pytorch.ipynb` has a stray `s` after an
  `imshow(...)` call in the image-display cell; it will cause a syntax error
  when that cell runs.
- The current playground notebook has user changes in progress. Preserve them
  unless a session specifically asks to modify it.

## Update rule

Refresh this file after a substantial session with: current focus, newly
confirmed understanding, the exact next action, and any important blocker or
code note. Keep it under one screen.
