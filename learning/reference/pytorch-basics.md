# PyTorch basics — quick reference

## The first debugging check

```python
print(x.shape, x.dtype, x.device)
```

For model inputs, also check `x.requires_grad` when gradients are relevant.

## Common shapes

| Meaning | Shape |
| --- | --- |
| One grayscale MNIST image | `(1, 28, 28)` |
| Batch of 4 grayscale MNIST images | `(4, 1, 28, 28)` |
| Batch of 4 RGB images | `(4, 3, 28, 28)` |
| Labels for a batch of 4 classification examples | `(4,)` |
| Dense-layer inputs | `(batch, input_features)` |
| Class logits | `(batch, num_classes)` |
| Token embeddings | `(batch, tokens, embedding_dim)` |

`N` (or “batch”) is the number of independent examples processed together.

## Create tensors

```python
torch.zeros(2, 3)                         # float32 zeros by default
torch.ones((2, 3), dtype=torch.int64)      # integer ones
torch.arange(0, 12, 2, dtype=torch.float32)
torch.rand(4, 1, 28, 28)                   # uniform values in [0, 1)
torch.randn(2, 3)                          # normal-distribution values
```

## Shape operations

```python
x.reshape(3, 8)            # returns a reshaped tensor
x = x.reshape(3, 8)        # keep the returned tensor
x.flatten(start_dim=1)     # preserve batch; flatten each example
x.squeeze()                # remove all size-1 dimensions
x.squeeze(1)               # remove dimension 1 only if it has size 1
x.transpose(1, 2)          # swap dimensions 1 and 2
```

For plotting a single grayscale image from `(1, 28, 28)`, use
`image.squeeze()` to give `imshow` a 2D `(28, 28)` array.

## Matrix multiplication

```python
features = torch.randn(3, 4)   # 3 examples, 4 input features each
weights = torch.randn(4, 2)    # map 4 input features to 2 outputs
outputs = features @ weights   # shape: (3, 2)
```

The inner dimensions must agree: `(a, **b**) @ (**b**, c) → (a, c)`.

## Matplotlib: figure and subplot

```python
plt.figure(figsize=(6, 3))  # one 6 × 3 inch canvas
plt.subplot(2, 2, 1)        # select position 1 in a 2-row × 2-column grid
plt.imshow(image, cmap="gray")
plt.title("Label: 3")
plt.axis("off")
plt.tight_layout()
```

A **figure** is the overall canvas. A **subplot** is one plotting area inside
the canvas. `plt.subplot(3, 2, i + 1)` reserves six positions; a four-item loop
leaves the third row empty.
