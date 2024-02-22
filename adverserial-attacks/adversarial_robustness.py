import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms

from PIL import Image

import logging

# Configure logging
import glob

log_files = glob.glob('log/*.log')

# create log file name with incremental version
log_file_name = 'log/adversarial_robustness_v'
if log_files:
    log_file_name += str(len(log_files) + 1)
log_file_name += '.log'

logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Replace print statements with log messages
logging.info('Your log message here')

# read the image, resize to 224 and convert to PyTorch Tensor
img_size = 224
pig_img = Image.open("adverserial-attacks/res/pig.jpg")
preprocess = transforms.Compose([
   transforms.Resize(img_size),
   transforms.ToTensor(),
])
pig_tensor = preprocess(pig_img)[None,:,:,:]

# plot image (note that numpy using HWC whereas Pytorch user CHW, so we need to convert)
# the input tensor has shape (1, 3, 224, 224), but matplotlib expects (height, width, channels)
# so we need to transpose to (224, 224, 3) before passing to imshow
# the first dimension is the batch dimension, but since we only have one image,
# we can ignore it
plt.imshow(pig_tensor[0].numpy().transpose(1,2,0))

plt.savefig('adverserial-attacks/out/pig_img.png', bbox_inches='tight')



# download the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='adverserial-attacks/data', train=True,
                                            download=True, transform=transforms.Compose([
                                                transforms.Resize(img_size),
                                                transforms.ToTensor()
                                            ]))
test_dataset = torchvision.datasets.MNIST(root='adverserial-attacks/data', train=False,
                                           download=True, transform=transforms.Compose([
                                               transforms.Resize(img_size),
                                               transforms.ToTensor()
                                           ]))

# create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# define model
class Net(torch.nn.Module):
    def __init__(self):
        """
        Constructor for the Net class.
        """
        super(Net, self).__init__()
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
        x = x.view(-1, img_size**2)  # flatten input image
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = self.fc2(x)
        return x

model = Net()

# train the model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(5):  # loop over the dataset multiple times
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

        if i % 100 == 0:
            logging.info('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch+1, 5, i+1, len(train_loader), loss.item()))

# test the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

logging.info('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))




