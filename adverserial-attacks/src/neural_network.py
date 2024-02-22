import logging
import torch

# define model
class NeuralNetwork(torch.nn.Module):
    def __init__(self, img_size):
        """
        Constructor for the Net class.
        """
        super(NeuralNetwork, self).__init__()
        self.image_size = img_size
        self.fc1 = torch.nn.Linear(img_size**2, 128)  # 784 -> 128
        self.fc2 = torch.nn.Linear(128, 10)  # 128 -> 10

    def forward(self, x):
        """
        Perform the forward pass of the neural network.

        Args:
            self: the object instance
            x: input data

        Returns:
            x: output of the neural network
        """
        x = x.view(-1, self.image_size**2)  # flatten input image
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = self.fc2(x)
        return x
    
def train_model(model, criterion, optimizer, train_loader, epochs):
    """
    Train the model.

    Args:
    - model: the neural network model
    - criterion: the loss function
    - optimizer: the optimization algorithm
    - train_loader: the data loader for training dataset
    - epochs: number of epochs to train

    Returns:
    - accuracy of the network on the training dataset
    """
    correct = 0
    total = 0
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 0:
                logging.info(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        accuracy = 100 * correct / total
        logging.info(f'Training Accuracy: {accuracy:.2f} %')
    return accuracy
    
def test_model(model, test_loader, dataset_name):
    """
    Test the model.

    Args:
    - model: the neural network model
    - test_loader: the data loader for test dataset

    Returns:
    - accuracy of the network on the test dataset
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    logging.info(f'Accuracy of the network on the {dataset_name} dataset: {accuracy:.2f} %')
    return accuracy