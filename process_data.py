import torch
from torchvision import datasets, transforms

def process_data(train_dir, test_dir, valid_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    data_test_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform = data_transforms)
    image_test_datasets = datasets.ImageFolder(test_dir, transform = data_test_transforms)
    image_valid_datasets = datasets.ImageFolder(train_dir, transform = data_test_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(image_test_datasets, batch_size=64)
    valid_dataloaders = torch.utils.data.DataLoader(image_valid_datasets, batch_size=64)
    
    return dataloaders, test_dataloaders, valid_dataloaders