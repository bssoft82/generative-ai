from matplotlib import pyplot as plt
from torchvision import transforms
from cnn import create_data_loaders, check_and_load_model 
from logging_config import setup_logging
import torch
import logging
from logging_config import log_entry
from fgsm import adversarial_attack, denorm
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50

setup_logging()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
epsilons = [0, .1, .2, .3]

def test( model, device, test_loader, epsilon, attack_method = 'fgsm'):
    # Accuracy counter
    correct = 0
    target_correct = 0
    adv_examples = []
    target_adv_examples = []

    # get one random image from the dataset and use it as target
    random_index = torch.randint(0, len(test_loader.dataset), (1,)).item() - 1
    target_data, target_target_test = next(iter(test_loader))[0], next(iter(test_loader))[0]
    target_data, target_target_test = target_data.to(device), target_target_test.to(device)
    target_data.requires_grad = True
    # Forward pass the data through the model
    target_output = model(target_data)
    target_pred = target_output.max(1, keepdim=True)[1] # get the index of the max log-probability

    # Loop over all examples in test set
    for data, target in test_loader:

        # Calculate the loss
        target_loss = F.nll_loss(target_output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        target_loss.backward(retain_graph=True)

        # Collect ``datagrad``
        target_data_grad = target_data.grad.data

        # Restore the data to its original scale
        target_data_denorm = denorm(device, target_data)
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
        perturbed_data = adversarial_attack(data_denorm, epsilon, data_grad, None, attack_method)
        perturbed_data_with_target = adversarial_attack(target_data_denorm, epsilon, target_data_grad, target_data_denorm, attack_method)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        perturbed_data_with_target_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data_with_target)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)
        target_output = model(perturbed_data_with_target_normalized)

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

        # Check for success with targeted attack
        targeted_final_pred = target_output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if targeted_final_pred.item() == target.item():
            target_correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(target_adv_examples) < 5:
                targeted_adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                target_adv_examples.append( (target_pred.item(), final_pred.item(), targeted_adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(target_adv_examples) < 5:
                targeted_adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                target_adv_examples.append( (target_pred.item(), final_pred.item(), targeted_adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    final_target_acc = target_correct/float(len(test_loader))
    logging.info(f'UnTargeted - {attack_method}\tEpsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}')
    logging.info(f'Targeted - {attack_method}\tEpsilon: {epsilon}\tTest Accuracy = {target_correct} / {len(test_loader)} = {final_target_acc}')

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


fgsm_accuracies = []
fgsm_examples = []
pgd_accuracies = []
pgd_examples = []

# Run test for each epsilon
for eps in epsilons:
    #training using resnet50
    model = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.eval()
    acc, ex = test (model_net, device, test_loader, eps)
    fgsm_accuracies.append (acc)
    fgsm_examples.append (ex)    
    pgd_acc, pgd_ex = test (model_net, device, test_loader, eps, 'pgd')
    pgd_accuracies.append (pgd_acc)
    pgd_examples.append (pgd_ex) 

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(fgsm_examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(fgsm_examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
        orig,adv,ex = fgsm_examples[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
plt.savefig('adverserial-attacks/out/resnet50_fgsm_image_comparison.png', bbox_inches='tight')

cnt = 0
for i in range(len(epsilons)):
    for j in range(len(pgd_examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(pgd_examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
        orig,adv,ex = pgd_examples[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
plt.savefig('adverserial-attacks/out/resnet50_pgd_image_comparison.png', bbox_inches='tight')

#training using own model
# Run test for each epsilon
for eps in epsilons:
    acc, ex = test (model_net, device, test_loader, eps)
    fgsm_accuracies.append (acc)
    fgsm_examples.append (ex)    
    pgd_acc, pgd_ex = test (model_net, device, test_loader, eps, 'pgd')
    pgd_accuracies.append (pgd_acc)
    pgd_examples.append (pgd_ex) 

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(fgsm_examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(fgsm_examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
        orig,adv,ex = fgsm_examples[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
plt.savefig('adverserial-attacks/out/fgsm_image_comparison.png', bbox_inches='tight')

cnt = 0
for i in range(len(epsilons)):
    for j in range(len(pgd_examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(pgd_examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
        orig,adv,ex = pgd_examples[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
plt.savefig('adverserial-attacks/out/pgd_image_comparison.png', bbox_inches='tight')