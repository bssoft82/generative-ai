import torch
import torch.nn as nn

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        """
        Constructor for the Normalize class.

        Parameters:
            mean (Tensor): The mean value for normalization.
            std (Tensor): The standard deviation for normalization.
        """
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
   
    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Normalized input tensor after subtracting the mean and dividing by the standard deviation.
        """
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

