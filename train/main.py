# import os
import argparse
import warnings

import numpy as np
import torch
from torch import nn, optim

from models.cnn import CNN
from utils.load_data import load_data

# from . import learning_rate, num_epochs

warnings.filterwarnings('ignore')


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_args():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--learning_rate', type = float, default = 0.001)
    parser.add_argument('--num_epochs', type = int, default = 5)

    # Model parameters
    parser.add_argument('--model_path', type = str,
                        default = '../save/model.pth')

    # Other parameters
    parser.add_argument('--is_train', type = bool, default = True)
    parser.add_argument('--is_test', type = bool, default = True)
    parser.add_argument('--device', type = str, default = 'cuda')

    return parser.parse_args()


def train(model, train_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')


def test(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            is_correct = (predicted == labels)
            correct += is_correct.sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f'Model saved at {model_path}')


def main():
    args = get_args()

    device = torch.device(args.device)  # Set device

    train_loader, test_loader = load_data()

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

    if args.is_train:
        train(model, train_loader, criterion, optimizer, args.num_epochs, device)
        save_model(model, '../save/model.pth')

    if args.is_test:
        test(model, test_loader, device)


if __name__ == "__main__":
    main()
