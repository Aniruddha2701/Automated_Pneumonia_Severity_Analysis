from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from preprocessing.transforms import train_transform, val_test_transform


def get_dataloaders(batch_size=64):

    train_dataset = ImageFolder(
        "Dataset/Final/train",
        transform=train_transform
    )

    val_dataset = ImageFolder(
        "Dataset/Final/val",
        transform=val_test_transform
    )

    test_dataset = ImageFolder(
        "Dataset/Final/test",
        transform=val_test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader