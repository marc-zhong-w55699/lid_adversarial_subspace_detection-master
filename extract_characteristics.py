from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity
from torchvision import datasets, transforms
import numpy as np
from util import get_data, extract_data, get_model
from util import (get_data, get_noisy_samples, get_mc_predictions,
                  get_deep_representations, score_samples, normalize,
                  get_lids_random_batch, get_kmeans_random_batch)

# In the original paper, the author used optimal KDE bandwidths dataset-wise
# that were determined from CV tuning
BANDWIDTHS = {'mnist': 3.7926, 'cifar': 0.26, 'svhn': 1.00}

# Here we further tune bandwidth for each of the 10 classes in mnist, cifar and svhn
# Run tune_kernal_density.py to get the following settings.
# BANDWIDTHS = {'mnist': [0.2637, 0.1274, 0.2637, 0.2637, 0.2637, 0.2637, 0.2637, 0.2069, 0.3360, 0.2637],
#               'cifar': [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#               'svhn': [0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1274, 0.1000, 0.1000]}

PATH_DATA = "./data/"
PATH_IMAGES = "./plots/"

def merge_and_generate_labels(X_pos, X_neg):
    """
    Merge positive and negative artifacts and generate labels.
    :param X_pos: Positive samples
    :param X_neg: Negative samples
    :return: X: Merged samples, 2D tensor
             y: Generated labels (0/1), 2D tensor same size as X
    """
    X_pos = torch.tensor(X_pos, dtype=torch.float32)
    print("X_pos: ", X_pos.shape)
    X_pos = X_pos.view(X_pos.size(0), -1)

    X_neg = torch.tensor(X_neg, dtype=torch.float32)
    print("X_neg: ", X_neg.shape)
    X_neg = X_neg.view(X_neg.size(0), -1)

    X = torch.cat((X_pos, X_neg), dim=0)
    y = torch.cat((torch.ones(X_pos.size(0)), torch.zeros(X_neg.size(0))))
    y = y.view(X.size(0), 1)

    return X, y
def evaluate_model(model: nn.Module, 
                   x: torch.Tensor, 
                   y: torch.Tensor, 
                   batch_size: int) -> float:
    """Evaluate model accuracy on data."""
    x = x.to(dtype=torch.float32)
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
            
            # 根据 batch_y 的维度决定如何处理
            if batch_y.ndim == 2:  # 如果是独热编码
                correct += (predicted == batch_y.argmax(1)).sum().item()
            elif batch_y.ndim == 1:  # 如果是类别索引
                correct += (predicted == batch_y).sum().item()
            else:
                raise ValueError(f"Unexpected shape for batch_y: {batch_y.shape}")
            
            total += batch_y.size(0)
    
    return correct / total
def model_predict(model, X, batch_size=256, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Perform predictions on the given data.

    :param model: PyTorch model
    :param X: Input data as a PyTorch tensor
    :param batch_size: Number of samples per batch for prediction
    :param device: Device to use for prediction ('cuda' or 'cpu')
    :return: Predicted class indices as a NumPy array
    """
    model.to(device)  # Move model to the specified device
    model.eval()  # Set model to evaluation mode
    
    # Ensure input tensor is on the same device
    X = X.to(device)
    predictions = []

    # Disable gradient computation for prediction
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            outputs = model(batch)  # Forward pass
            # Convert logits to predicted class indices
            preds = torch.argmax(outputs, dim=1)
            predictions.append(preds.cpu())  # Move to CPU for concatenation
    predictions = torch.cat(predictions).numpy()
    # Combine predictions from all batches
    return predictions
def get_kd(model, X_train, Y_train, X_test, X_test_noisy, X_test_adv, batch_size):
    """
    Get kernel density scores.
    :param model: PyTorch model
    :param X_train: Training data
    :param Y_train: Training labels
    :param X_test: Test data
    :param X_test_noisy: Noisy test data
    :param X_test_adv: Adversarial test data
    :param batch_size: Batch size for processing
    :return: Artifacts: Positive and negative examples with KD values
             Labels: Adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    # Get deep feature representations
    print('Getting deep feature representations...')
    X_train_features = get_deep_representations(model, X_train, batch_size)
    X_test_normal_features = get_deep_representations(model, X_test, batch_size)
    X_test_noisy_features = get_deep_representations(model, X_test_noisy, batch_size)
    X_test_adv_features = get_deep_representations(model, X_test_adv, batch_size)

    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(Y_train.size(1)):
        class_inds[i] = torch.where(Y_train.argmax(dim=1) == i)[0]
    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined "
                  "optimal for the specific CNN models of the paper. If you've "
                  "changed your model, you'll need to re-optimize the "
                  "bandwidth.")
    print(f'Bandwidth {BANDWIDTHS[args.dataset]:.4f} for {args.dataset}')
    for i in range(Y_train.size(1)):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=BANDWIDTHS[args.dataset]) \
            .fit(X_train_features[class_inds[i]].numpy())

    # Get model predictions
    print('Computing model predictions...')
    preds_test_normal = torch.argmax(model(X_test), dim=1)
    preds_test_noisy = torch.argmax(model(X_test_noisy), dim=1)
    preds_test_adv = torch.argmax(model(X_test_adv), dim=1)

    # Get density estimates
    print('Computing densities...')
    densities_normal = score_samples(kdes, X_test_normal_features, preds_test_normal.numpy())
    densities_noisy = score_samples(kdes, X_test_noisy_features, preds_test_noisy.numpy())
    densities_adv = score_samples(kdes, X_test_adv_features, preds_test_adv.numpy())

    print("Densities_normal:", len(densities_normal))
    print("Densities_adv:", len(densities_adv))
    print("Densities_noisy:", len(densities_noisy))

    # Skip the normalization, you may want to try different normalizations later
    # At this step, just save the raw values
    densities_pos = torch.tensor(densities_adv)
    densities_neg = torch.cat((torch.tensor(densities_normal), torch.tensor(densities_noisy)))
    artifacts, labels = merge_and_generate_labels(densities_pos, densities_neg)

    return artifacts, labels

def get_bu(model, X_test, X_test_noisy, X_test_adv):
    """
    Get Bayesian uncertainty scores
    :param model: 
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param X_test_noisy: 
    :param X_test_adv: 
    :return: artifacts: positive and negative examples with bu values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Getting Monte Carlo dropout variance predictions...')
    uncerts_normal = get_mc_predictions(model, X_test,
                                        batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)
    uncerts_noisy = get_mc_predictions(model, X_test_noisy,
                                       batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)
    uncerts_adv = get_mc_predictions(model, X_test_adv,
                                     batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)

    print("uncerts_normal:", uncerts_normal.shape)
    print("uncerts_noisy:", uncerts_noisy.shape)
    print("uncerts_adv:", uncerts_adv.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z = normalize(
    #     uncerts_normal,
    #     uncerts_adv,
    #     uncerts_noisy
    # )

    uncerts_pos = uncerts_adv
    uncerts_neg = np.concatenate((uncerts_normal, uncerts_noisy))
    artifacts, labels = merge_and_generate_labels(uncerts_pos, uncerts_neg)

    return artifacts, labels

def get_lid(model, X_test, X_test_noisy, X_test_adv,device, k=10, batch_size=100):
    """
    Get local intrinsic dimensionality
    :param model: 
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param X_test_noisy: 
    :param X_test_adv: 
    :return: artifacts: positive and negative examples with lid values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Extract local intrinsic dimensionality: k = %s' % k)
    lids_normal, lids_noisy, lids_adv = get_lids_random_batch(model, X_test, X_test_noisy, X_test_adv,device, k,batch_size)
    print("lids_normal:", lids_normal.shape)
    print("lids_noisy:", lids_noisy.shape)
    print("lids_adv:", lids_adv.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # lids_normal_z, lids_adv_z, lids_noisy_z = normalize(
    #     lids_normal,
    #     lids_adv,
    #     lids_noisy
    # )

    lids_pos = lids_adv
    lids_neg = np.concatenate((lids_normal, lids_noisy))
    artifacts, labels = merge_and_generate_labels(lids_pos, lids_neg)

    return artifacts, labels

def get_kmeans(model, X_test, X_test_noisy, X_test_adv, k=10, batch_size=100, dataset='mnist'):
    """
    Calculate the average distance to k nearest neighbours as a feature.
    This is used to compare density vs LID. Why density doesn't work?
    :param model: 
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param X_test_noisy: 
    :param X_test_adv: 
    :return: artifacts: positive and negative examples with lid values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Extract k means feature: k = %s' % k)
    kms_normal, kms_noisy, kms_adv = get_kmeans_random_batch(model, X_test, X_test_noisy,
                                                              X_test_adv, dataset, k, batch_size,
                                                             pca=True)
    print("kms_normal:", kms_normal.shape)
    print("kms_noisy:", kms_noisy.shape)
    print("kms_adv:", kms_adv.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # kms_normal_z, kms_noisy_z, kms_adv_z = normalize(
    #     kms_normal,
    #     kms_noisy,
    #     kms_adv
    # )

    kms_pos = kms_adv
    kms_neg = np.concatenate((kms_normal, kms_noisy))
    artifacts, labels = merge_and_generate_labels(kms_pos, kms_neg)

    return artifacts, labels

def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw-l2'"
    assert args.characteristic in ['kd', 'bu', 'lid', 'km', 'all'], \
        "Characteristic(s) to use 'kd', 'bu', 'lid', 'km', 'all'"
    model_file = os.path.join(PATH_DATA, "model_%s.pth" % args.dataset)
    assert os.path.isfile(model_file), \
        'model file not found... must first train model using train_model.py.'
    adv_file = os.path.join(PATH_DATA, "Adv_%s_%s.pt" % (args.dataset, args.attack))
    assert os.path.isfile(adv_file), \
        'adversarial sample file not found... must first craft adversarial ' \
        'samples using craft_adv_samples.py'

    print('Loading the data and model...')
     # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model
    
    if args.attack in ['cw-l2', 'cw-lid']:
        warnings.warn("Important: remove the softmax layer for cw attacks!")
        model = get_model(args.dataset, softmax=False)
    else:
        model = get_model(args.dataset)
    
    model.load_state_dict(torch.load(model_file))
    model = model.to(device)
    model.eval()
    # Load the dataset
    train_loader, test_loader  = get_data(args.dataset,args.batch_size)
    # 提取数据
    X_train, Y_train, X_test, Y_test = extract_data(train_loader, test_loader)
    # Check attack type, select adversarial and noisy samples accordingly
    print('Loading noisy and adversarial samples...')
    if args.attack == 'all':
        # TODO: implement 'all' option
        # X_test_adv = ...
        # X_test_noisy = ...
        raise NotImplementedError("'All' types detector not yet implemented.")
    else:
        # Load adversarial samples
        X_test_adv = torch.load(adv_file)
        print("X_test_adv: ", X_test_adv.shape)

        # as there are some parameters to tune for noisy example, so put the generation
        # step here instead of the adversarial step which can take many hours
        noisy_file = os.path.join(PATH_DATA, 'Noisy_%s_%s.npy' % (args.dataset, args.attack))
        if os.path.isfile(noisy_file):
            X_test_noisy = np.load(noisy_file)
        else:
            # Craft an equal number of noisy samples
            print('Crafting %s noisy samples. ' % args.dataset)
            X_test_noisy = get_noisy_samples(X_test, X_test_adv, args.dataset, args.attack)
            np.save(noisy_file, X_test_noisy)

    # Check model accuracies on each sample type
    for s_type, data in zip(['normal', 'noisy', 'adversarial'],
                               [X_test, X_test_noisy, X_test_adv]):
        
        acc = evaluate_model(model, data, Y_test, args.batch_size)
        print("Model accuracy on the %s test set: %0.2f%%" %
              (s_type, 100 * acc))
        # Compute and display average perturbation sizes
        if not s_type == 'normal':
            l2_diff = np.linalg.norm(
                data.reshape((len(data), -1)) -
                X_test.reshape((len(X_test), -1)),
                axis=1
            ).mean()
            print("Average L-2 perturbation size of the %s test set: %0.2f" %
                  (s_type, l2_diff))

    # Refine the normal, noisy and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    preds_test = model_predict(model, X_test, batch_size=args.batch_size, device=device)
    inds_correct = np.where(preds_test == Y_test.argmax(axis=1))[0]
    print("Number of correctly predict images: %s" % (len(inds_correct)))

    X_test = X_test[inds_correct]
    X_test_noisy = X_test_noisy[inds_correct]
    X_test_adv = X_test_adv[inds_correct]
    print("X_test: ", X_test.shape)
    print("X_test_noisy: ", X_test_noisy.shape)
    print("X_test_adv: ", X_test_adv.shape)

    if args.characteristic == 'kd':
        # extract kernel density
        characteristics, labels = get_kd(model, X_train, Y_train, X_test, X_test_noisy, X_test_adv)
        print("KD: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)

        # save to file
        bandwidth = BANDWIDTHS[args.dataset]
        file_name = os.path.join(PATH_DATA, 'kd_%s_%s_%.4f.npy' % (args.dataset, args.attack, bandwidth))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
    elif args.characteristic == 'bu':
        # extract Bayesian uncertainty
        characteristics, labels = get_bu(model, X_test, X_test_noisy, X_test_adv)
        print("BU: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)

        # save to file
        file_name = os.path.join(PATH_DATA, 'bu_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
    elif args.characteristic == 'lid':
        # extract local intrinsic dimensionality
      
        characteristics, labels = get_lid(model, X_test, X_test_noisy, X_test_adv,device,
                                    args.k_nearest, args.batch_size)
        print("LID: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)

        # save to file
        # file_name = os.path.join(PATH_DATA, 'lid_%s_%s.npy' % (args.dataset, args.attack))
        file_name = os.path.join('../data_grid_search/lid_large_batch/', 'lid_%s_%s_%s.npy' %
                                 (args.dataset, args.attack, args.k_nearest))

        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
    elif args.characteristic == 'km':
        # extract k means distance
        characteristics, labels = get_kmeans(model, X_test, X_test_noisy, X_test_adv,
                                    args.k_nearest, args.batch_size, args.dataset)
        print("K-Mean: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)

        # save to file
        file_name = os.path.join(PATH_DATA, 'km_pca_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
    elif args.characteristic == 'all':
        # extract kernel density
        characteristics, labels = get_kd(model, X_train, Y_train, X_test, X_test_noisy, X_test_adv)
        file_name = os.path.join(PATH_DATA, 'kd_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)

        # extract Bayesian uncertainty
        characteristics, labels = get_bu(model, X_test, X_test_noisy, X_test_adv)
        file_name = os.path.join(PATH_DATA, 'bu_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)

        # extract local intrinsic dimensionality
        characteristics, labels = get_lid(model, X_test, X_test_noisy, X_test_adv,
                                    args.k_nearest, args.batch_size, args.dataset)
        file_name = os.path.join(PATH_DATA, 'lid_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)

        # extract k means distance
        # artifcharacteristics, labels = get_kmeans(model, X_test, X_test_noisy, X_test_adv,
        #                                args.k_nearest, args.batch_size, args.dataset)
        # file_name = os.path.join(PATH_DATA, 'km_%s_%s.npy' % (args.dataset, args.attack))
        # data = np.concatenate((characteristics, labels), axis=1)
        # np.save(file_name, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'jsma', 'bim-b', 'jsma', 'cw-l2' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-r', '--characteristic',
        help="Characteristic(s) to use 'kd', 'bu', 'lid' 'km' or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-k', '--k_nearest',
        help="The number of nearest neighbours to use; either 10, 20, 100 ",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=100)
    parser.set_defaults(k_nearest=20)
    args = parser.parse_args()
    main(args)
