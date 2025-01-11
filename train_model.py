import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from util import get_data, get_model


# Training function
def train(dataset='mnist', batch_size=128, epochs=50):
    print(f"Dataset: {dataset}")
    train_loader, test_loader = get_data(dataset,batch_size)
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
