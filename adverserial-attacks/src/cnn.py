import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
from logging_config import log_entry


# Define the CNN model class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Config for constants
CNN_CONFIG = {
    "print_interval": 100,
    "sub_interval": 2,
    "test_loss_divisor": 1,
    "fc_input_size": 320,
    "model_path": "adverserial-attacks/models/cnn_model.pth"
}

# Function to check if saved model exists and load it
@log_entry
def check_and_load_model(device, train_loader, test_loader,train_again = False, epochs=10):
    model_path = CNN_CONFIG["model_path"]
    model = Net().to(device)
    if os.path.isfile(model_path) and not train_again:
        logging.info(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logging.info("Training a new model")
        train_model(model, train_loader, test_loader, epochs)
        torch.save(model.state_dict(), model_path)

    model.eval()
    return model

# Function to create data loaders
@log_entry
def create_data_loaders(root_folder, dataset_name, train_transform, test_transform, batch_size=64):
    train_dataset = getattr(datasets, dataset_name)(root=root_folder, train=True, download=True, transform=train_transform)
    test_dataset = getattr(datasets, dataset_name)(root=root_folder, train=False, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CNN_CONFIG["test_loss_divisor"], shuffle=False)
    return train_loader, test_loader

# Function to train the model
@log_entry
def train_model(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % CNN_CONFIG["print_interval"] == 0:
                accuracy = 100 * correct / total
                logging.info(f'Epoch {epoch + 1} iteration {i + 1}: Loss {running_loss:.3f} Accuracy {accuracy:.2f} %')

        train_losses.append(running_loss / CNN_CONFIG["print_interval"])
        train_accuracies.append(accuracy)
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            test_losses.append(criterion(outputs, labels).item())
            test_accuracies.append(accuracy)
            logging.info(f'Epoch {epoch+1}: Test Loss {test_losses[-1]:.3f} Test Accuracy {accuracy:.2f}%')

    # Save the model after training
    model_path = CNN_CONFIG["model_path"]
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")

    # Save the plots
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('adverserial-attacks/out/model_loss.png', bbox_inches='tight')

    plt.clf()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('adverserial-attacks/out/model_accuracy.png', bbox_inches='tight')
    


