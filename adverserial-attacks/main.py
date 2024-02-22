import logging
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from PIL import Image

from logging_config import setup_logging
from neural_network import NeuralNetwork, test_model, train_model

# Setup logging
setup_logging()

def get_image_tensor(image_path, image_size):
    # read the image, resize to image_size and convert to PyTorch Tensor
    pig_img = Image.open(image_path)
    preprocess = transforms.Compose([
       transforms.Resize(image_size),
       transforms.ToTensor(),
    ])
    return preprocess(pig_img)[None,:,:,:]

def load_data (root_folder, image_size):
    # download the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=root_folder, train=True,
                                                download=True, transform=transforms.Compose([
                                                    transforms.Resize(image_size),
                                                    transforms.ToTensor()
                                                ]))
    test_dataset = torchvision.datasets.MNIST(root=root_folder, train=False,
                                            download=True, transform=transforms.Compose([
                                                transforms.Resize(image_size),
                                                transforms.ToTensor()
                                            ]))

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader







# Main program logic
if __name__ == "__main__":
    image_size = 224

    pig_tensor = get_image_tensor("adverserial-attacks/res/pig.jpg", image_size)
    plt.imshow(pig_tensor[0].numpy().transpose(1,2,0))
    plt.savefig('adverserial-attacks/out/pig_img.png', bbox_inches='tight')

    train_data, test_data, train_loader, test_loader = load_data (
        'adverserial-attacks/data', 
        image_size=224)
    
    model = NeuralNetwork(image_size)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimization algorithm
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, epochs=5)

    # Test the model
    train_accuracy = test_model(model, train_loader, "training")
    test_accuracy = test_model(model, test_loader, "test")

    # Compare the accuracy
    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    


