# Curated sources

Use these as anchors when a topic needs a precise API detail or a deeper,
structured treatment. They are intentionally few.

## Primary path

1. [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) — follow selectively alongside the local playground: tensors, data loading, model building, autograd, optimization, and saving/loading.
2. [PyTorch autograd notes](https://docs.pytorch.org/docs/stable/notes/autograd.html) — consult when computation-graph or gradient behavior is unclear.
3. [CrossEntropyLoss reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — use as the definitive API reference while learning logits and classification targets.
4. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — read after the attention playground is familiar; focus first on the attention equations and architecture diagram.

## Local learning path

- `../mnist-from-scratch.ipynb` — dataset-to-NumPy-network foundations.
- `../main.py` — a complete manual forward/backprop/update pass.
- `../mnist-using-pytorch.ipynb` — an end-to-end MNIST MLP in PyTorch.
- `../pytorch_hands_on_playground.ipynb` — the current experiment-first bridge to CNNs, embeddings, and attention.

## Selection rule

Use official documentation for exact or changing framework behavior. Use the
local notebooks for conceptual experiments. Do not add material merely because
it is popular; add it when it unlocks the current roadmap.
