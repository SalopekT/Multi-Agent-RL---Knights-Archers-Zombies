import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from AngleNet import AngleNet
import matplotlib.pyplot as plt


def evaluate(model, test_loader, device):
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.normalize(output, dim=1)
            target = F.normalize(target, dim=1)
            all_outputs.append(output)
            all_targets.append(target)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # cosine similarity -> angle error in degrees
    angle_error = torch.acos(F.cosine_similarity(all_outputs, all_targets).clamp(-1,1))
    angle_error = angle_error.mean() * 180 / torch.pi
    return angle_error.item()



def main():
    # Load first test dataset
    data0 = np.load("../dataset_angles/dataset_test3.npz")
    images0 = data0['images'].astype(np.float32)   # (N,41,41,3)
    labels0 = data0['labels'].astype(np.float32)   # (N,2)

    # Normalize images to [-1,1]
    images0 = (images0 / 255.0 - 0.5) / 0.5

    # Convert to tensors and permute
    images0 = torch.tensor(images0).permute(0,3,1,2)
    labels0 = torch.tensor(labels0)
    labels0 = F.normalize(labels0, dim=1)

    # Create DataLoader
    test_dataset0 = TensorDataset(images0, labels0)
    test_loader0 = DataLoader(test_dataset0, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AngleNet().to(device)
    model.load_state_dict(torch.load("anglenet.pth"))
    model.eval()   # very important for evaluation

    error0 = evaluate(model, test_loader0, device)
    print(f"Dataset 0 average angle error: {error0:.2f}°")

    num_examples = 20
    example_images = images0[350:370]
    example_labels = labels0[350:370]
    model.eval()
    with torch.no_grad():
        example_images = example_images.to(device)
        preds = model(example_images)
        preds = F.normalize(preds, dim=1)

        # convert cosine similarity to angle in degrees
        angle_errors = torch.acos(F.cosine_similarity(preds, example_labels[:num_examples].to(device)).clamp(-1,1))
        angle_errors = angle_errors * 180 / torch.pi
    for i in range(num_examples):
        print(f"Example {i+1}:")
        print(f"  Predicted vector: {preds[i].cpu().numpy()}")
        print(f"  Target vector:    {example_labels[i].cpu().numpy()}")
        print(f"  Angle error:      {angle_errors[i].item():.2f}°\n")

    for i in range(num_examples):
        img = example_images[i].cpu().permute(1,2,0).numpy()  # CHW -> HWC
        img = (img * 0.5) + 0.5  # convert back from [-1,1] to [0,1] for display
        plt.imshow(img)
        plt.title(f"Pred vs Target: {angle_errors[i].item():.2f}°")
        plt.axis('off')
        plt.show()
if __name__ == "__main__":
    main()