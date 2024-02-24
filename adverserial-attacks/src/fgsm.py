import torch

# FGSM attack code
def adversarial_attack(image, epsilon, data_grad, target_image=None, attack_algo='fgsm'):
    # Create the perturbed image by adjusting each pixel of the input image
    if attack_algo == 'fgsm':
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        if target_image is not None:
            perturbed_image = target_image - epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    else:
        perturbed_image = image + epsilon * torch.sign(data_grad - image + 0.5)
        if target_image is not None:
            perturbed_image += image + epsilon * torch.sign(data_grad - target_image + 0.5)
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
    # Return the perturbed image
    return perturbed_image

# restores the tensors to their original scale
def denorm(device, batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)