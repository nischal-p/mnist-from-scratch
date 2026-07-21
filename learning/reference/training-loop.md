# Training loop — quick reference

```python
model.train()
for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)

    predictions = model(inputs)       # forward pass
    loss = loss_fn(predictions, targets)

    optimizer.zero_grad()             # clear accumulated gradients
    loss.backward()                   # calculate gradients
    optimizer.step()                  # update parameters
```

Evaluation does not update parameters:

```python
model.eval()
with torch.no_grad():
    predictions = model(inputs)
    loss = loss_fn(predictions, targets)
```

## Classification with `CrossEntropyLoss`

```python
logits = model(images)                 # (batch, classes), raw scores
targets = labels                       # (batch,), integer class IDs
loss = nn.CrossEntropyLoss()(logits, targets)
```

Do not put `softmax` in the model immediately before `CrossEntropyLoss`; use
`logits.softmax(dim=1)` only when inspecting probabilities.

## Common failures

| Symptom | Check |
| --- | --- |
| Matrix-size error | shapes before the failing layer |
| "Expected Long" / target error | classification targets should be integer class IDs |
| Device mismatch | move model and every batch tensor to the same device |
| Gradients unexpectedly grow | ensure `optimizer.zero_grad()` occurs before `backward()` |
| Test results vary | call `model.eval()` and use `torch.no_grad()` |
