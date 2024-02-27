from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_data(batch_size = 64):
    """
    This function is used to load the MNIST dataset for both training and testing.
    Parameters:
        batch_size (int): The number of samples per batch to load. Default is 64.

    Returns:
        tuple: Returns a tuple containing the DataLoader instances for the
    training and testing datasets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root = './data', train = True,
                                               download = True,
                                               transform = transform)
    test_dataset = datasets.MNIST(root = './data', train = False,
                                              download = True,
                                              transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = batch_size,
                              shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size,
                             shuffle = False)

    return train_loader, test_loader
