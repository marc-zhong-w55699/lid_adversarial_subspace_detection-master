import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define a simple function to get data
def get_data(dataset, batch_size):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    elif dataset == 'cifar':
        transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar'.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Define a simple model
def get_model(dataset):
    if dataset == 'mnist':
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    elif dataset == 'cifar':
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar'.")
    return model

# Training function
def train(dataset='mnist', batch_size=128, epochs=50):
    print(f"Dataset: {dataset}")
    train_loader, test_loader = get_data(dataset, batch_size)
    model = get_model(dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), f"data/model_{dataset}.pth")

# Main function
def main(args):
    assert args.dataset in ['mnist', 'cifar', 'all'], \
        "Dataset parameter must be either 'mnist', 'cifar', or 'all'"
    
    if args.dataset == 'all':
        for dataset in ['mnist', 'cifar']:
            train(dataset, args.batch_size, args.epochs)
    else:
        train(args.dataset, args.batch_size, args.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help="Dataset to use; either 'mnist', 'cifar', or 'all'", required=True, type=str)
    parser.add_argument('-e', '--epochs', help="The number of epochs to train for.", required=False, type=int, default=50)
    parser.add_argument('-b', '--batch_size', help="The batch size to use for training.", required=False, type=int, default=128)
    args = parser.parse_args()
    main(args)
