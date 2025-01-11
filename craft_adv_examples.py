import os
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, Optional

from util import get_data, get_model , extract_test_data
from attacks import fgsm, basic_iterative_method, saliency_map_method
from cw_attacks import CarliniL2, CarliniLID  # assuming these are converted to PyTorch

# Attack parameters
ATTACK_PARAMS = {
    'mnist': {'eps': 0.40, 'eps_iter': 0.010, 'image_size': 28, 'num_channels': 1, 'num_labels': 10},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005, 'image_size': 32, 'num_channels': 3, 'num_labels': 10},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010, 'image_size': 32, 'num_channels': 3, 'num_labels': 10}
}

CLIP_MIN = -0.5
CLIP_MAX = 0.5
PATH_DATA = "./data/"

def evaluate_model(model: nn.Module, 
                  x: torch.Tensor, 
                  y: torch.Tensor, 
                  batch_size: int) -> float:
    """Evaluate model accuracy on data."""
    model.eval()
    device = next(model.parameters()).device
    dataloader = DataLoader(list(zip(x, y)), batch_size=batch_size)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            correct += (predicted == batch_y.argmax(1)).sum().item()
            total += batch_y.size(0)
    
    return correct / total

def compute_l2_diff(x_adv: torch.Tensor, x_orig: torch.Tensor) -> float:
    """Compute average L2 perturbation size."""
    diff = (x_adv - x_orig).view(len(x_adv), -1)
    l2_diff = torch.norm(diff, p=2, dim=1).mean().item()
    return l2_diff

def craft_one_type(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   dataset: str,
                   attack: str,
                   batch_size: int) -> None:
    """
    Craft adversarial examples using specified attack method.
    
    Args:
        model: target model
        x: input samples
        y: true labels
        dataset: dataset name ('mnist', 'cifar', 'svhn')
        attack: attack type ('fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2', 'cw-lid')
        batch_size: batch size for attacks
    """
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    
    if attack == 'fgsm':
        print('Crafting fgsm adversarial samples...')
        x_adv = fgsm(
            model=model,
            x=x,
            y=y,
            eps=ATTACK_PARAMS[dataset]['eps'],
            clip_min=CLIP_MIN,
            clip_max=CLIP_MAX
        )
    
    elif attack in ['bim-a', 'bim-b']:
        print(f'Crafting {attack} adversarial samples...')
        its, results = basic_iterative_method(
            model=model,
            x=x,
            y=y,
            eps=ATTACK_PARAMS[dataset]['eps'],
            eps_iter=ATTACK_PARAMS[dataset]['eps_iter'],
            nb_iter=50,
            clip_min=CLIP_MIN,
            clip_max=CLIP_MAX
        )
        
        if attack == 'bim-a':
            # Select first misclassified step for each sample
            x_adv = torch.stack([results[its[i]][i] for i in range(len(y))])
        else:
            # Select last step for all samples
            x_adv = results[-1]
    
    elif attack == 'jsma':
        print('Crafting jsma adversarial samples. This may take > 5 hours')
        x_adv = saliency_map_method(
            model=model,
            x=x,
            y=y,
            theta=1,
            gamma=0.1,
            clip_min=CLIP_MIN,
            clip_max=CLIP_MAX
        )
    
    elif attack == 'cw-l2':
        print('Crafting C&W L2 examples. This takes > 5 hours due to internal grid search')
        image_size = ATTACK_PARAMS[dataset]['image_size']
        num_channels = ATTACK_PARAMS[dataset]['num_channels']
        num_labels = ATTACK_PARAMS[dataset]['num_labels']
        cw_attack = CarliniL2(model, image_size, num_channels, num_labels, batch_size)
        x_adv = cw_attack.attack(x, y)
    
    elif attack == 'cw-lid':
        print('Crafting C&W LID examples. This takes > 5 hours due to internal grid search')
        image_size = ATTACK_PARAMS[dataset]['image_size']
        num_channels = ATTACK_PARAMS[dataset]['num_channels']
        num_labels = ATTACK_PARAMS[dataset]['num_labels']
        cw_attack = CarliniLID(model, image_size, num_channels, num_labels, batch_size)
        x_adv = cw_attack.attack(x, y)
    
    # Evaluate and save results
    acc = evaluate_model(model, x_adv, y, batch_size)
    print(f"Model accuracy on the adversarial test set: {100 * acc:.2f}%")
    
    # Save adversarial examples
    save_path = os.path.join(PATH_DATA, f'Adv_{dataset}_{attack}.pt')
    torch.save(x_adv.cpu(), save_path)
    
    # Compute and print L2 perturbation size
    l2_diff = compute_l2_diff(x_adv, x)
    print(f"Average L-2 perturbation size of the {attack} attack: {l2_diff:.2f}")

def main(args):
    # Validate arguments
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2', 'all', 'cw-lid'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2', 'all' or 'cw-lid'"
    
    model_file = os.path.join(PATH_DATA, f"model_{args.dataset}.pth")
    assert os.path.isfile(model_file), \
        'Model file not found... must first train model using train_model.py'
    
    if args.dataset == 'svhn' and args.attack == 'cw-l2':
        assert args.batch_size == 16, \
            "SVHN has 26032 test images, batch_size for cw-l2 attack should be 16"
    
    print(f'Dataset: {args.dataset}. Attack: {args.attack}')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if args.attack in ['cw-l2', 'cw-lid']:
        warnings.warn("Important: remove the softmax layer for cw attacks!")
        model = get_model(args.dataset, softmax=False)
    else:
        model = get_model(args.dataset)
    
    model.load_state_dict(torch.load(model_file))
    model = model.to(device)
    model.eval()
    
    # Load test data
    train_loader, test_loader  = get_data(args.dataset,args.batch_size)
    x_test, y_test = extract_test_data(test_loader)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    # Evaluate clean accuracy
    acc = evaluate_model(model, x_test, y_test, args.batch_size)
    print(f"Accuracy on the test set: {100 * acc:.2f}%")
    
    # Handle CW-LID case
    if args.attack == 'cw-lid':
        x_test = x_test[:1000]
        y_test = y_test[:1000]
    
    # Craft adversarial examples
    if args.attack == 'all':
        for attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2']:
            craft_one_type(model, x_test, y_test, args.dataset, attack, args.batch_size)
    else:
        craft_one_type(model, x_test, y_test, args.dataset, args.attack, args.batch_size)
    
    print(f'Adversarial samples crafted and saved to {PATH_DATA}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2' or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int,
        default=100
    )
    args = parser.parse_args()
    main(args)
