import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
#https://github.com/pytorch/examples/blob/main/mnist/main.py
class AngleNet(nn.Module):
    def __init__(self):
        super(AngleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(12800, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)  # if needed

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.tanh(x) 
        x = F.normalize(x, dim=1)

        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        #https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html
        loss = 1-F.cosine_similarity(output, target).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*data.size(0)
    epoch_loss = epoch_loss/len(train_loader.dataset)

    print(f"Average loss in epoch: {epoch_loss:.4f}")
    model.train()



def main():
    print("Hello")
    images_list = []
    targets_list = []
    for i in range(6):
        data = np.load(f"../dataset_angles/dataset{i}.npz")
        images = data['images']  
        labels = data['labels']
        images_list.append(images)
        targets_list.append(labels)
    images = np.concatenate(images_list, axis=0)
    images = images/255.0
    images = (images-0.5)/0.5
    labels = np.concatenate(targets_list, axis=0)
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    labels = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AngleNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    for epoch in range(1, num_epochs+1):
        train(model, device, train_loader, optimizer, epoch)

    torch.save(model.state_dict(), "anglenet2.pth")
    print("Model saved!")

if __name__ == "__main__":
    main()