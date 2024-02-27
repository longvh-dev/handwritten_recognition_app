# import os
import warnings

import numpy as np
import torch
from torch import nn, optim

from models.cnn import CNN
from utils.load_data import load_data

# from . import learning_rate, num_epochs

warnings.filterwarnings('ignore')


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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

            correct += np.sum(predicted == labels).item()

    accuracy = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f'Model saved at {model_path}')


def main():
    learning_rate = 0.001
    num_epochs = 5
    device = torch.device('cuda')  # Set device to GPU

    train_loader, test_loader = load_data()

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    train(model, train_loader, criterion, optimizer, num_epochs, device)
    save_model(model, '../save/model.pth')
    test(model, test_loader, device)


if __name__ == "__main__":
    main()
