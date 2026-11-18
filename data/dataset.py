from torchvision import datasets, transforms
import numpy as np


def get_transform(normalize=True):
    """Returns the preprocessing transformation of CIFAR10 dataset."""

    if not normalize:
        return transforms.ToTensor()
    
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
            ),
        ]
    )


def load_cifar10(root="./data", normalize=True):
    """Downloads the CIFAR10 dataset. Returns train and test datsets"""

    transform = get_transform(normalize)

    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset

def print_dataset_distribution(dataset):
    """Prints the dataset class distribution"""

    distribution = np.zeros(len(dataset.classes), dtype=int)

    for i in range(len(dataset)):
        distribution[dataset.targets[i]]+=1

    for j in range(len(dataset.classes)):
        print(dataset.classes[j] + ": " + str(distribution[j]) + " (" + str(distribution[j]/len(dataset)*100) + "%)")