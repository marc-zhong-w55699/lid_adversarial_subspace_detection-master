import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple
import copy

def fgsm(model: nn.Module,
         x: torch.Tensor,
         eps: float,
         clip_min: Optional[float] = None,
         clip_max: Optional[float] = None,
         y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    PyTorch implementation of the Fast Gradient Sign Method (FGSM).
    
    Args:
        model: the model to attack
        x: the input tensor
        eps: the epsilon (input variation parameter)
        clip_min: minimum value for components of the example returned
        clip_max: maximum value for components of the example returned
        y: the label tensor. Use None to avoid label leaking effect
    
    Returns:
        adversarial example tensor
    """
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    
    # Get model predictions
    outputs = model(x_adv)
    
    # If no labels provided, use model predictions
    if y is None:
        _, y = outputs.max(1)
        y = F.one_hot(y, outputs.shape[1]).float()
    
    # Calculate loss
    loss = F.cross_entropy(outputs, y.argmax(dim=1))
    
    # Get gradient
    loss.backward()
    grad = x_adv.grad.data
    
    # Create adversarial example
    x_adv = x_adv + eps * grad.sign()
    
    # Clip if needed
    if clip_min is not None or clip_max is not None:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    
    return x_adv.detach()

def jsma(model: nn.Module,
         x: torch.Tensor,
         target: int,
         theta: float,
         gamma: float,
         clip_min: Optional[float] = None,
         clip_max: Optional[float] = None,
         verbose: bool = False) -> Tuple[torch.Tensor, int, float]:
    """
    PyTorch implementation of the Jacobian-based Saliency Map Attack (JSMA).
    
    Args:
        model: the model to attack
        x: the input tensor (single sample)
        target: target class
        theta: delta for each feature adjustment
        gamma: maximum distortion percentage (between 0 and 1)
        clip_min: minimum value for components of the example
        clip_max: maximum value for components of the example
        verbose: whether to print progress
    
    Returns:
        tuple of (adversarial example, success flag, percent perturbed)
    """
    model.eval()
    device = x.device
    
    # Initialize
    adv_x = x.clone().detach()
    nb_features = torch.prod(torch.tensor(x.shape[1:])).item()
    max_iters = int(np.floor(nb_features * gamma / 2))
    
    if verbose:
        print(f'Maximum number of iterations: {max_iters}')
    
    # Define search domain
    if clip_max is not None:
        search_domain = set((adv_x < clip_max).nonzero().cpu().numpy().flatten())
    else:
        search_domain = set(range(nb_features))
    
    # Main loop
    iteration = 0
    current = model(adv_x.unsqueeze(0)).argmax().item()
    
    while current != target and iteration < max_iters and len(search_domain) > 1:
        # Calculate Jacobian
        adv_x.requires_grad_(True)
        outputs = model(adv_x.unsqueeze(0))
        
        # Get gradients for all classes
        jacobian = torch.zeros((outputs.shape[1], *adv_x.shape)).to(device)
        for class_idx in range(outputs.shape[1]):
            if class_idx == target:
                continue
            outputs[0, class_idx].backward(retain_graph=True)
            jacobian[class_idx] = adv_x.grad.clone()
            adv_x.grad.zero_()
            
        # Get target gradients
        outputs[0, target].backward()
        jacobian[target] = adv_x.grad.clone()
        adv_x.grad.zero_()
        adv_x.requires_grad_(False)
        
        # Compute saliency map
        target_grad = jacobian[target].flatten()
        other_grad = jacobian[torch.arange(jacobian.shape[0]) != target].sum(0).flatten()
        
        # Select two pixels to modify
        valid_pixels = torch.tensor(list(search_domain), device=device)
        saliency_map = (target_grad[valid_pixels] * abs(other_grad[valid_pixels]))
        
        if len(valid_pixels) < 2:
            break
            
        # Get top two pixels
        values, indices = saliency_map.topk(2)
        pixel_idx = valid_pixels[indices].cpu().numpy()
        
        # Apply perturbation
        adv_x_flat = adv_x.flatten()
        adv_x_flat[pixel_idx] += theta
        if clip_max is not None:
            adv_x_flat[pixel_idx] = torch.clamp(adv_x_flat[pixel_idx], clip_min, clip_max)
        adv_x = adv_x_flat.reshape(adv_x.shape)
        
        # Update search domain
        search_domain.remove(pixel_idx[0])
        search_domain.remove(pixel_idx[1])
        
        # Update loop variables
        current = model(adv_x.unsqueeze(0)).argmax().item()
        iteration += 1
        
        if verbose and iteration % 5 == 0:
            print(f'Current iteration: {iteration} - Current Prediction: {current}')
    
    # Calculate perturbation percentage
    percent_perturbed = float(iteration * 2) / nb_features
    success = 1 if current == target else 0
    
    if verbose:
        print('Successful' if success else 'Unsuccessful')
    
    return adv_x, success, percent_perturbed

def basic_iterative_method(model: nn.Module,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          eps: float,
                          eps_iter: float,
                          nb_iter: int = 50,
                          clip_min: Optional[float] = None,
                          clip_max: Optional[float] = None) -> Tuple[dict, torch.Tensor]:
    """
    PyTorch implementation of the Basic Iterative Method (BIM).
    
    Args:
        model: the model to attack
        x: the input tensor
        y: the label tensor
        eps: maximum distortion
        eps_iter: attack step size
        nb_iter: number of iterations
        clip_min: minimum value for components of the example
        clip_max: maximum value for components of the example
    
    Returns:
        tuple of (iteration counts dict, adversarial examples)
    """
    model.eval()
    batch_size = x.shape[0]
    
    # Initialize
    x_adv = x.clone().detach()
    x_min = x_adv - eps
    x_max = x_adv + eps
    
    # Track iterations for each sample
    its = {i: nb_iter-1 for i in range(batch_size)}
    out = set()
    
    # Store results
    results = torch.zeros((nb_iter, *x.shape), device=x.device)
    
    print('Running BIM iterations...')
    for i in tqdm(range(nb_iter)):
        x_adv = fgsm(model, x_adv, eps_iter, clip_min, clip_max, y)
        x_adv = torch.max(torch.min(x_adv, x_max), x_min)
        results[i] = x_adv
        
        # Check misclassified samples
        predictions = model(x_adv).argmax(dim=1)
        misclassified = (predictions != y.argmax(dim=1)).nonzero().flatten()
        
        for idx in misclassified:
            idx = idx.item()
            if idx not in out:
                its[idx] = i
                out.add(idx)
    
    return its, results


def saliency_map_method(model: nn.Module,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       theta: float,
                       gamma: float,
                       clip_min: Optional[float] = None,
                       clip_max: Optional[float] = None) -> torch.Tensor:
    """
    PyTorch implementation of the Saliency Map Method attack.
    
    Args:
        model: the model to attack
        x: the input tensor
        y: the label tensor
        theta: perturbation size
        gamma: maximum percentage of perturbed features
        clip_min: minimum value for components of the example
        clip_max: maximum value for components of the example
    
    Returns:
        tensor of adversarial examples
    """
    model.eval()
    device = x.device
    nb_classes = y.shape[1]
    
    # Initialize adversarial examples
    x_adv = torch.zeros_like(x, device=device)
    
    def other_classes(nb_classes: int, current_class: int) -> np.ndarray:
        """Returns a list of class indices excluding the current class."""
        return np.array([i for i in range(nb_classes) if i != current_class])
    
    # Generate adversarial examples for each input
    for i in tqdm(range(len(x))):
        current_class = int(torch.argmax(y[i]))
        target_class = int(np.random.choice(other_classes(nb_classes, current_class)))
        
        # Apply JSMA
        x_adv[i], _, _ = jsma(
            model=model,
            x=x[i],
            target=target_class,
            theta=theta,
            gamma=gamma,
            clip_min=clip_min,
            clip_max=clip_max
        )
    
    return x_adv
