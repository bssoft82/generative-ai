from matplotlib import pyplot as plt
from torchvision import transforms
from cnn import create_data_loaders, check_and_load_model 
from logging_config import setup_logging
import torch
import logging
from logging_config import log_entry
from fgsm import fgsm_attack, denorm
import torch.nn.functional as F

setup_logging()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
epsilons = [0, .05, .1, .15, .2, .25, .3]

def test( model, device, test_loader, epsilon ):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(device, data)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    logging.info("Epsilon: %s\tTest Accuracy = %d / %d = %f", epsilon, correct, len(test_loader), final_acc)

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

ADV_ATTACK_CONFIG = {
    "data_folder":"adverserial-attacks/data",
    "dataset_name": "MNIST",
    "epochs": 10
}

train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_loader, test_loader = create_data_loaders(ADV_ATTACK_CONFIG['data_folder'], ADV_ATTACK_CONFIG['dataset_name'], train_transform, test_transform, batch_size = 1)

model_net = check_and_load_model(device, train_loader, test_loader, train_again = False, epochs = ADV_ATTACK_CONFIG['epochs'])


accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test (model_net, device, test_loader, eps)
    accuracies.append (acc)
    examples.append (ex)    

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
plt.savefig('adverserial-attacks/out/image_comparison.png', bbox_inches='tight')