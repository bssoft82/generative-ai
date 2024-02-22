import logging
import matplotlib
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from img_normalize import Normalize
from torchvision import transforms
from torchvision.models import resnet50
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

def pred_own_nn_model():
    model = NeuralNetwork(image_size)
    model.load_state_dict(torch.load('./models/model.pth'))
    model.eval()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimization algorithm
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, epochs=1)

    # Test the model
    train_accuracy = test_model(model, train_loader, "training")
    test_accuracy = test_model(model, test_loader, "test")

    # Compare the accuracy
    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")

    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pred = model (norm(test_data))
    return pred

def pred_resnet50_model():
    # values are standard normalization for ImageNet images, 
    # from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # load pre-trained ResNet50, and put into evaluation mode (necessary to e.g. turn off batchnorm)
    model = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.eval()
    # form predictions
    pred = model(norm(pig_tensor))
    logging.info(f'Predictions from resnet50 model: {pred}')

    return pred

# Main program logic
if __name__ == "__main__":
    image_size = 224

    pig_tensor = get_image_tensor("adverserial-attacks/res/pig.jpg", image_size)
    plt.imshow(pig_tensor[0].numpy().transpose(1,2,0))
    plt.savefig('adverserial-attacks/out/pig_img.png', bbox_inches='tight')

    train_data, test_data, train_loader, test_loader = load_data (
        'adverserial-attacks/data', 
        image_size)
    
    pred_own_nn = pred_own_nn_model()
    pred_resnet50 = pred_resnet50_model ()

    with open("adverserial-attacks/res/imagenet_class_index.json") as f:
        imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}

    logging.info(f'ResNet50 prediction class: {imagenet_classes[pred_resnet50.max(dim=1)[1].item()]}')
    logging.info(f'Affinity to target class: {nn.CrossEntropyLoss()(pred_resnet50,torch.LongTensor([341])).item()}')
        
    logging.info(f'Own NN prediction class: {imagenet_classes[pred_own_nn.max(dim=1)[1].item()]}')
    logging.info(f'Affinity to target class: {nn.CrossEntropyLoss()(pred_own_nn,torch.LongTensor([341])).item()}')


