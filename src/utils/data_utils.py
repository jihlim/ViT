import os

import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

def get_input_image_size(dataset_name):
    if dataset_name.lower() == "mnist":
        i_size = 224
    elif dataset_name.lower() == "cifar10":
        i_size = 224
    elif dataset_name.lower() == "imagenet1k":
        i_size = 224
    return i_size

def get_dataset(data_root, dataset_name, i_size):
    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((i_size, i_size)),
            transforms.RandomCrop(i_size, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((i_size, i_size)),
            transforms.ToTensor()
        ]
    )

    # Datasets
    if dataset_name.lower() == "mnist":
        trainset = datasets.MNIST(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        testset = datasets.MNIST(
            root=data_root,
            train=False,
            transform=test_transform,
            download=True,
        )
        num_classes = 10
        
    elif dataset_name.lower() == "cifar10":
        trainset = datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        testset = datasets.CIFAR10(
            root=data_root,
            train=False,
            transform=test_transform,
            download=True,
        )
        num_classes = 10
        
    elif dataset_name.lower() == "imagenet1k":
        trainset = datasets.ImageNet(
            root=os.path.join(data_root, "imagenet1k"),
            split="train",
            transform=train_transform,
        )
        testset = datasets.ImageNet(
            root=os.path.join(data_root, "imagenet1k"),
            split="val",
            transform=test_transform,
        )
        num_classes = 1000
    
    return trainset, testset, num_classes

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(src_dir)
    print(root_dir)