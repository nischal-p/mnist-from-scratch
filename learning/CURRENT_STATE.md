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
- Broadcasting can reuse a `(3,)` offset across each row of a `(2, 3)` tensor.
- `mean(dim=1)` produces one value per row, shape `(2,)`; `keepdim=True`
  instead yields `(2, 1)`, so it can broadcast back across the columns.
- Dense-layer matrix multiplication follows `(batch, input_features) @
  (input_features, output_features) -> (batch, output_features)`.

## Best next session

Run the reorganized early tensor-fluency path in
`notebooks/02-pytorch-playground.ipynb`: tensor facts → image axes → indexing
and reshaping → combining and summarizing tensors. Predict the printed shapes
before running each section.

1. Complete the row-wise squared-length exercise and state its input/output
   shapes: `(2, 2) -> (2,)`.
2. Continue the playground's **Indexing, reshaping, and views** section:
   predict each `flatten(start_dim=...)` result and assign
   `my_var = my_var.reshape(3, 8)`.
3. Proceed to autograd only after these shape operations feel predictable.

## Known code notes (do not fix automatically)

- `notebooks/03-mnist-using-pytorch.ipynb` has a stray `s` after an
  `imshow(...)` call in the image-display cell; it will cause a syntax error
  when that cell runs.
- The current playground notebook has user changes in progress. Preserve them
  unless a session specifically asks to modify it.
- The revised broadcasting section has static JSON validation, but was not
  executed here because the `jupyter` command is not installed in this workspace.

## Update rule

Refresh this file after a substantial session with: current focus, newly
confirmed understanding, the exact next action, and any important blocker or
code note. Keep it under one screen.
