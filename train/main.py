import os
import warnings

import torch
from torch import nn, optim
import numpy as np

from train.models.cnn import CNN
from train.utils.load_data import load_data
from . import learning_rate, num_epochs

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train(model, train_loader, criterion, optimizer, epochs = 5):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch [{epoch + 1} / {num_epochs}], Step [{i + 1}'
                      f'/{len(train_loader)}], Loss: {loss.item():.4f}')

                running_loss = 0.0

    print('Finished Training')


def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += np.sum(predicted == labels).item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f'Model saved at {model_path}')

def main():
    train_loader, test_loader = load_data()

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    train(model, train_loader, criterion, optimizer, num_epochs)
    save_model(model, 'save/model.pth')
    test(model, test_loader)


if __name__ == "__main__":
    main()
