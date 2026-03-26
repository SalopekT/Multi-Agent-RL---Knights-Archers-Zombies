import numpy as np
data = np.load("dataset_angles/dataset2.npz")
images = data["images"]
labels = data["labels"]

print(images.shape)
print(labels.shape)

print(labels[:10])

angles = np.arctan2(labels[:,1], labels[:,0])
print(np.histogram(angles, bins=8))