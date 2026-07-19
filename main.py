import numpy as np
from datasets import load_dataset


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative_from_activation(a):
    # If a = sigmoid(z), then sigmoid'(z) = a * (1 - a)
    return a * (1 - a)


# ------------------------------------------------------------
# 1. Load one MNIST example
# ------------------------------------------------------------

dataset = load_dataset("ylecun/mnist")
example = dataset["train"][0]

image = example["image"]
label = example["label"]

# 28 × 28 image → 784 inputs
# Pixel values: 0–255 → 0–1
inputs = np.array(image, dtype=np.float64).flatten() / 255.0

# Convert the label into a 10-value target vector.
target = np.zeros(10)
target[label] = 1


# ------------------------------------------------------------
# 2. Initialize the network
#
# 784 inputs
#   → 16 hidden neurons
#   → 10 output neurons
# ------------------------------------------------------------

rng = np.random.default_rng(seed=0)

hidden_weights = rng.normal(
    loc=0,
    scale=0.1,
    size=(16, 784),
)

hidden_bias = np.zeros(16)

output_weights = rng.normal(
    loc=0,
    scale=0.1,
    size=(10, 16),
)

output_bias = np.zeros(10)


# ------------------------------------------------------------
# 3. Forward pass
# ------------------------------------------------------------

hidden_z = hidden_weights @ inputs + hidden_bias
hidden_activations = sigmoid(hidden_z)

output_z = output_weights @ hidden_activations + output_bias
output_activations = sigmoid(output_z)

prediction_before = np.argmax(output_activations)

loss_before = np.mean(
    (target - output_activations) ** 2
)

print("Actual label:", label)
print("Prediction before update:", prediction_before)
print("Output activations:", output_activations)
print("Loss before update:", loss_before)


# ------------------------------------------------------------
# 4. Backpropagation: output layer
# ------------------------------------------------------------

# Our loss is:
#
# mean((target - output)^2)
#
# Since there are 10 output neurons:
#
# d_loss / d_output
# =
# 2 * (output - target) / 10

num_outputs = len(output_activations)

d_loss_d_output = (
    2
    * (output_activations - target)
    / num_outputs
)

# The output passed through sigmoid:
#
# output_activations = sigmoid(output_z)
#
# Use the chain rule to find:
#
# d_loss / d_output_z

d_loss_d_output_z = (
    d_loss_d_output
    * sigmoid_derivative_from_activation(
        output_activations
    )
)


# ------------------------------------------------------------
# 5. Gradients for output weights and biases
# ------------------------------------------------------------

# d_loss_d_output_z has shape:
# (10,)
#
# hidden_activations has shape:
# (16,)
#
# We need one gradient for every output-layer weight:
# (10, 16)

d_loss_d_output_weights = np.outer(
    d_loss_d_output_z,
    hidden_activations,
)

# Each output neuron has one bias. 
# Since output_z = weight @ hidden_activations + bias,
# here, changing the bias by 1 changes the output_z by 1, so
d_loss_d_output_bias = d_loss_d_output_z


# ------------------------------------------------------------
# 6. Propagate the error into the hidden layer
# ------------------------------------------------------------

# output_weights:
# (10, 16)
#
# output_weights.T:
# (16, 10)
#
# d_loss_d_output_z:
# (10,)
#
# Result:
# (16,)
#
# This tells us how the loss changes with respect
# to each hidden activation.

d_loss_d_hidden = (
    output_weights.T @ d_loss_d_output_z
)

# The hidden activations also came from sigmoid,
# so apply the chain rule again.

hidden_delta = (
    d_loss_d_hidden
    * sigmoid_derivative_from_activation(
        hidden_activations
    )
)


# ------------------------------------------------------------
# 7. Gradients for hidden weights and biases
# ------------------------------------------------------------

# hidden_delta has shape:
# (16,)
#
# inputs has shape:
# (784,)
#
# We need one gradient for every hidden-layer weight:
# (16, 784)

hidden_weights_gradient = np.outer(
    hidden_delta,
    inputs,
)

# Each hidden neuron has one bias.
hidden_bias_gradient = hidden_delta


# ------------------------------------------------------------
# 8. Update every weight and bias once
# ------------------------------------------------------------

learning_rate = 0.1

output_weights -= (
    learning_rate
    * d_loss_d_output_weights
)

output_bias -= (
    learning_rate
    * d_loss_d_output_bias
)

hidden_weights -= (
    learning_rate
    * hidden_weights_gradient
)

hidden_bias -= (
    learning_rate
    * hidden_bias_gradient
)


# ------------------------------------------------------------
# 9. Run the same image again after the update
# ------------------------------------------------------------

hidden_z = hidden_weights @ inputs + hidden_bias
hidden_activations = sigmoid(hidden_z)

output_z = output_weights @ hidden_activations + output_bias
output_activations = sigmoid(output_z)

prediction_after = np.argmax(output_activations)

loss_after = np.mean(
    (target - output_activations) ** 2
)

print()
print("Prediction after update:", prediction_after)
print("Output activations:", output_activations)
print("Loss after update:", loss_after)
print("Loss decreased:", loss_after < loss_before)