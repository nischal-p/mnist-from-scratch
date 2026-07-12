from datasets import load_dataset
import numpy as np

dataset = load_dataset("ylecun/mnist")

example = dataset["train"][0]

image = example["image"]
label = example["label"]

pixels = np.array(image)

print(pixels[0][0])
print(pixels[14][14])