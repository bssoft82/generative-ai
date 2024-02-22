import logging
import os
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

def pred_own_nn_model(norm, train_again = False):
    model_path = "adverserial-attacks/models/model.pth"
    if os.path.isfile(model_path) and not train_again:
        print(f"Loading existing model from {model_path}")
        model = NeuralNetwork(image_size)
        model.load_state_dict(torch.load(model_path))
    else:
        print("Training a new model")
        model = NeuralNetwork(image_size)
        # Define the loss function
        criterion = nn.CrossEntropyLoss()

        # Define the optimization algorithm
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Train the model
        train_model(model, criterion, optimizer, train_loader, epochs=10)

        # Save the model
        torch.save(model.state_dict(), model_path)

    model.eval()

    # Test the model
    train_accuracy = test_model(model, train_loader, "training")
    test_accuracy = test_model(model, test_loader, "test")

    # Compare the accuracy
    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")

    pred = model (norm(pig_tensor))
    return model, pred

def pred_resnet50_model(norm):
    # load pre-trained ResNet50, and put into evaluation mode (necessary to e.g. turn off batchnorm)
    model = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.eval()
    # form predictions
    pred = model(norm(pig_tensor))
    logging.info(f'Predictions from resnet50 model: {pred}')

    return model, pred

def perform_adversarial_attack(attack_method, attack_type, image_tensor, delta, model, norm, orig_img_idx, tgt_img_idx, epsilon = 2./255):
    if attack_method == 'fgsm':
        pred = model(norm(image_tensor + delta))
        if attack_type == 'untargeted':
            loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([orig_img_idx]))
        elif attack_type == 'targeted':
             loss = (-nn.CrossEntropyLoss()(pred, torch.LongTensor([orig_img_idx])) + 
                    nn.CrossEntropyLoss()(pred, torch.LongTensor([tgt_img_idx])))
        loss.backward()
        logging.info(f'{attack_method} - {attack_type} : loss: {loss.item()}') # print loss.item())
        
        return epsilon * delta.grad.detach().sign()
    
    #elif attack_method == 'pgd':
    opt = optim.SGD([delta], lr=1e-1)

    for t in range(100):
        pred = model(norm(image_tensor + delta))
        if attack_type == 'untargeted':
            loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([orig_img_idx]))
        elif attack_type == 'targeted':
             loss = (-nn.CrossEntropyLoss()(pred, torch.LongTensor([orig_img_idx])) + 
                    nn.CrossEntropyLoss()(pred, torch.LongTensor([tgt_img_idx])))
        if t % 10 == 0:
            logging.info(f'{attack_method} - {attack_type} :Iteration {t} loss: {loss.item()}') # print loss.item())
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        delta.data.clamp_(-epsilon, epsilon)
    
    max_class = pred.max(dim=1)[1].item()
    logging.info(f'Predicted class: {imagenet_classes[max_class]}')
    logging.info(f'True class probability: {nn.Softmax(dim=1)(pred)[0,max_class].item()}')

# Main program logic
if __name__ == "__main__":
    image_size = 224

    pig_tensor = get_image_tensor("adverserial-attacks/res/pig.jpg", image_size)
    plt.imshow(pig_tensor[0].numpy().transpose(1,2,0))
    plt.savefig('adverserial-attacks/out/orig_pig_img.png', bbox_inches='tight')

    train_data, test_data, train_loader, test_loader = load_data (
        'adverserial-attacks/data', 
        image_size)
    
    # values are standard normalization for ImageNet images, 
    # from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model_own_nn, pred_own_nn = pred_own_nn_model (norm, train_again=False)
    model_resnet50, pred_resnet50 = pred_resnet50_model (norm)

    with open("adverserial-attacks/res/imagenet_class_index.json") as f:
        imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}

    logging.info(f'Predictions from own NN model: {pred_own_nn}')
    logging.info(f'Predictions from resnet50 model: {pred_resnet50}')
        
    # Note: the model returns log probabilities, so we need to take an exp to get
    #       the probabilities themselves.
    logging.info(f'Size of pred_own_nn: {pred_own_nn.size()}')
    logging.info(f'Size of pred_resnet50: {pred_resnet50.size()}')
    pred_resnet50 = torch.exp(pred_resnet50) # Take exp to get probabilities, not log probabilities

    # Resize pred_own_nn to match pred_resnet50
    pred_own_nn = pred_own_nn.unsqueeze(0)
    logging.info(f'Size of pred_own_nn after resizing: {pred_own_nn.size()}')

    logging.info(f"Own NN prediction: {imagenet_classes[pred_own_nn.argmax().item()]}")
    logging.info(f"ResNet50 prediction: {imagenet_classes[pred_resnet50.argmax().item()]}")

    logging.info(f"Own NN probability: {pred_own_nn.max().item():.2f}")
    logging.info(f"ResNet50 probability: {pred_resnet50.max().item():.2f}")

    logging.info(f"Own NN - ResNet50: {pred_own_nn.max().item() - pred_resnet50.max().item():.2f}")

    delta = torch.zeros_like(pig_tensor, requires_grad=True)
    perform_adversarial_attack ('fgsm', 'untargeted', pig_tensor, delta, model_resnet50, norm, 341, 404)
    plt.imshow((pig_tensor + delta)[0].detach().numpy().transpose(1,2,0))
    plt.savefig('adverserial-attacks/out/fgsm_untargeted_pig_img.png', bbox_inches='tight')
    plt.imshow((50*delta+0.5)[0].detach().numpy().transpose(1,2,0))
    plt.savefig('adverserial-attacks/out/delta_pig_img.png', bbox_inches='tight')
    
    perform_adversarial_attack ('fgsm', 'targeted', pig_tensor, delta, model_resnet50, norm, 341, 404)
    plt.imshow((pig_tensor + delta)[0].detach().numpy().transpose(1,2,0))
    plt.savefig('adverserial-attacks/out/fgsm_targeted_pig_img.png', bbox_inches='tight')

    perform_adversarial_attack ('pgd', 'untargeted', pig_tensor, delta, model_resnet50, norm, 341, 404)
    plt.imshow((pig_tensor + delta)[0].detach().numpy().transpose(1,2,0))
    plt.savefig('adverserial-attacks/out/pgd_targeted_pig_img.png', bbox_inches='tight')

    perform_adversarial_attack ('pgd', 'targeted', pig_tensor, delta, model_resnet50, norm, 341, 404)
    plt.imshow((pig_tensor + delta)[0].detach().numpy().transpose(1,2,0))
    plt.savefig('adverserial-attacks/out/pgd_targeted_pig_img.png', bbox_inches='tight')
