import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.decomposition import PCA


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets, transforms
from scipy.io import loadmat
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
# Constants
STDEVS = {
    'mnist': {'fgsm': 0.271, 'bim-a': 0.111, 'bim-b': 0.167, 'cw-l2': 0.207},
    'cifar': {'fgsm': 0.0504, 'bim-a': 0.0084, 'bim-b': 0.0428, 'cw-l2': 0.007},
    'svhn': {'fgsm': 0.133, 'bim-a': 0.0155, 'bim-b': 0.095, 'cw-l2': 0.008}
}
CLIP_MIN = -0.5
CLIP_MAX = 0.5
DATA_PATH = "data/"

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

def get_data(dataset, batch_size):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    elif dataset == 'cifar':
        transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar'.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def extract_test_data(dataset,test_loader):
    """
    提取测试数据并转换为适合模型输入的格式。
    支持 MNIST、CIFAR-10 和 SVHN 数据集。

    Args:
        test_loader (DataLoader): 测试集数据加载器。
        dataset_name (str): 数据集名称，支持 'mnist'、'cifar'、'svhn'。

    Returns:
        x_test (torch.Tensor): 测试数据张量。
        y_test (torch.Tensor): 测试标签张量（独热编码形式）。
    """
    x_test = []
    y_test = []
    
    for batch_x, batch_y in test_loader:
        x_test.append(batch_x)
        y_test.append(batch_y)
    
    # 合并批次数据为完整张量
    x_test = torch.cat(x_test, dim=0)
    y_test = torch.cat(y_test, dim=0)
    
    # 针对不同数据集进行标签预处理
    if dataset_name.lower() == 'svhn':
        # SVHN 数据集中，标签 10 表示数字 0，需要转换为 0
        y_test = y_test % 10
    elif dataset_name.lower() not in ['mnist', 'cifar']:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use 'mnist', 'cifar', or 'svhn'.")
    
    # 独热编码
    num_classes = torch.max(y_test) + 1  # 自动推断类别数量
    y_test = F.one_hot(y_test, num_classes=num_classes).float()
    
    return x_test, y_test

def get_model(dataset='mnist', softmax=True):
    """
    构建 PyTorch 模型，支持 MNIST、CIFAR 和 SVHN 数据集。
    :param dataset: 数据集名称（'mnist', 'cifar', 或 'svhn'）
    :param softmax: 是否添加 softmax 到最后一层
    :return: PyTorch 模型实例
    """
    assert dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset 参数必须是 'mnist', 'cifar', 或 'svhn'"

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            if dataset == 'mnist':
                # MNIST 模型
                self.features = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, padding=0),  # 28x28 -> 26x26
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.Conv2d(64, 64, kernel_size=3),  # 26x26 -> 24x24
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(kernel_size=2),  # 24x24 -> 12x12
                    nn.Dropout(0.5)
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 12 * 12, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.5),
                    nn.Linear(128, 10)
                )
            elif dataset == 'cifar':
                # CIFAR-10 模型
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x32 -> 32x32
                    nn.ReLU(),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32x32 -> 32x32
                    nn.ReLU(),
                    nn.BatchNorm2d(32),
                    nn.MaxPool2d(kernel_size=2),  # 32x32 -> 16x16

                    nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16 -> 16x16
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 16x16 -> 16x16
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(kernel_size=2),  # 16x16 -> 8x8

                    nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 8x8 -> 8x8
                    nn.ReLU(),
                    nn.BatchNorm2d(128),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 8x8 -> 8x8
                    nn.ReLU(),
                    nn.BatchNorm2d(128),
                    nn.MaxPool2d(kernel_size=2)  # 8x8 -> 4x4
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(128 * 4 * 4, 1024),
                    nn.ReLU(),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.5),
                    nn.Linear(512, 10)
                )
            else:
                # SVHN 模型
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=0),  # 32x32 -> 30x30
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.Conv2d(64, 64, kernel_size=3),  # 30x30 -> 28x28
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(kernel_size=2),  # 28x28 -> 14x14
                    nn.Dropout(0.5)
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 14 * 14, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.5),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.5),
                    nn.Linear(128, 10)
                )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            if softmax:
                x = F.softmax(x, dim=1)
            return x

    # 返回实例化的模型
    return Model()


def cross_entropy(y_true, y_pred):
    """
    y_true: Tensor of shape (batch_size, num_classes) (one-hot encoded labels)
    y_pred: Tensor of shape (batch_size, num_classes) (raw logits)
    """
    # Convert one-hot labels to class indices
    y_true_indices = torch.argmax(y_true, dim=1)
    # Compute cross-entropy loss
    return F.cross_entropy(y_pred, y_true_indices)

def lid_term(logits, batch_size=100):
    """
    Calculate LID loss term for a minibatch of logits

    :param logits: Tensor of shape (batch_size, num_classes)
    :return: LID values for the minibatch
    """
    # Assuming logits are already softmaxed
    y_pred = logits

    # Calculate pairwise distance
    r = torch.sum(torch.square(y_pred), dim=1)
    r = r.view(-1, 1)  # Turn r into column vector
    D = r - 2 * torch.matmul(y_pred, y_pred.T) + r.T

    # Find the k nearest neighbors
    D1 = torch.sqrt(D + 1e-9)  # Add epsilon for numerical stability
    D2, _ = torch.topk(-D1, k=21, dim=1, largest=False, sorted=True)
    D3 = -D2[:, 1:]  # Exclude the nearest neighbor (self)

    m = (D3.T / D3[:, -1]).T  # Normalize distances by the furthest neighbor
    v_log = torch.sum(torch.log(m + 1e-9), dim=1)  # Avoid NaN
    lids = -20 / v_log

    ## Batch normalize lids (optional)
    # lids = F.normalize(lids, p=2, dim=0)

    return lids

def lid_adv_term(clean_logits, adv_logits, batch_size=100):
    """
    Calculate LID loss term for a minibatch of adversarial logits.

    :param clean_logits: Tensor of clean logits (batch_size, num_features)
    :param adv_logits: Tensor of adversarial logits (batch_size, num_features)
    :return: LID values for the minibatch
    """
    # Reshape logits to ensure proper dimensions
    c_pred = clean_logits.view(batch_size, -1)
    a_pred = adv_logits.view(batch_size, -1)

    # Calculate pairwise distance
    r_a = torch.sum(torch.square(a_pred), dim=1).view(-1, 1)  # Column vector
    r_c = torch.sum(torch.square(c_pred), dim=1).view(1, -1)  # Row vector
    D = r_a - 2 * torch.matmul(a_pred, c_pred.T) + r_c

    # Find the k nearest neighbors
    D1 = torch.sqrt(D + 1e-9)  # Add epsilon for numerical stability
    D2, _ = torch.topk(-D1, k=21, dim=1, largest=False, sorted=True)  # k=21 includes self
    D3 = -D2[:, 1:]  # Exclude the nearest neighbor (self)

    # Normalize distances and calculate LID
    m = (D3.T / D3[:, -1]).T  # Normalize distances by the furthest neighbor
    v_log = torch.sum(torch.log(m + 1e-9), dim=1)  # Avoid NaN
    lids = -20 / v_log

    # Batch normalize lids (optional)
    lids = F.normalize(lids, p=2, dim=0)

    return lids

def flip(x, nb_diff):
    """
    Helper function for get_noisy_samples
    :param x:
    :param nb_diff:
    :return:
    """
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < CLIP_MAX)[0]
    assert candidate_inds.shape[0] >= nb_diff
    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = CLIP_MAX

    return np.reshape(x, original_shape)


def get_noisy_samples(X_test, X_test_adv, dataset, attack):
    """
    TODO
    :param X_test:
    :param X_test_adv:
    :param dataset:
    :param attack:
    :return:
    """
    if attack in ['jsma', 'cw-l0']:
        X_test_noisy = np.zeros_like(X_test)
        for i in range(len(X_test)):
            # Count the number of pixels that are different
            nb_diff = len(np.where(X_test[i] != X_test_adv[i])[0])
            # Randomly flip an equal number of pixels (flip means move to max
            # value of 1)
            X_test_noisy[i] = flip(X_test[i], nb_diff)
    else:
        warnings.warn("Important: using pre-set Gaussian scale sizes to craft noisy "
                      "samples. You will definitely need to manually tune the scale "
                      "according to the L2 print below, otherwise the result "
                      "will inaccurate. In future scale sizes will be inferred "
                      "automatically. For now, manually tune the scales around "
                      "mnist: L2/20.0, cifar: L2/54.0, svhn: L2/60.0")
        # Add Gaussian noise to the samples
        # print(STDEVS[dataset][attack])
        X_test_noisy = np.minimum(
            np.maximum(
                X_test + np.random.normal(loc=0, scale=STDEVS[dataset][attack],
                                          size=X_test.shape),
                CLIP_MIN
            ),
            CLIP_MAX
        )

    return X_test_noisy


def get_mc_predictions(model, X, nb_iter=50, batch_size=256):
    """
    使用 PyTorch 模型获取 Monte Carlo (MC) 预测。
    :param model: PyTorch 模型。
    :param X: 输入数据，形状为 (N, ...) 的 NumPy 数组。
    :param nb_iter: MC 采样的迭代次数。
    :param batch_size: 批大小。
    :return: MC 预测结果，形状为 (nb_iter, N, output_dim) 的 NumPy 数组。
    """
    # 确保模型处于训练模式 (启用 Dropout 等)
    model.train()

    # 输出维度
    device = next(model.parameters()).device  # 模型所用设备
    X = torch.tensor(X, dtype=torch.float32).to(device)  # 将输入数据转为 Tensor 并移动到设备上
    output_dim = model(torch.rand_like(X[:1])).shape[-1]  # 推测输出维度

    def predict():
        """对整个输入数据进行一次前向传播，返回批次预测值。"""
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        output = torch.zeros((len(X), output_dim), device=device)
        for i in range(n_batches):
            batch_X = X[i * batch_size:(i + 1) * batch_size]
            output[i * batch_size:(i + 1) * batch_size] = model(batch_X)
        return output.cpu().numpy()

    # 进行多次 MC 采样
    preds_mc = []
    for _ in tqdm(range(nb_iter), desc="MC Sampling"):
        preds_mc.append(predict())

    return np.asarray(preds_mc)




def get_deep_representations(model, X, batch_size=256):
    """
    获取深度表示（从指定隐藏层提取特征）。
    :param model: PyTorch 模型。
    :param X: 输入数据，形状为 (N, ...) 的 NumPy 数组。
    :param batch_size: 批大小。
    :return: 提取的深度表示，形状为 (N, output_dim) 的 NumPy 数组。
    """
    # 确保模型处于评估模式
    model.eval()

    # 设备信息
    device = next(model.parameters()).device
    X = torch.tensor(X, dtype=torch.float32).to(device)  # 将输入数据转为 Tensor 并移动到设备上

    # 找到倒数第4层
    layers = list(model.children())
    target_layer = layers[-4]

    # 定义从倒数第4层获取输出的辅助函数
    class RepresentationExtractor(torch.nn.Module):
        def __init__(self, base_model, target_layer):
            super().__init__()
            self.features = torch.nn.Sequential(*list(base_model.children())[:layers.index(target_layer) + 1])

        def forward(self, x):
            return self.features(x)

    extractor = RepresentationExtractor(model, target_layer).to(device)

    # 计算输出维度
    with torch.no_grad():
        sample_output = extractor(X[:1])
        output_dim = sample_output.shape[-1]

    # 批次处理
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros((len(X), output_dim))
    with torch.no_grad():
        for i in range(n_batches):
            batch_X = X[i * batch_size:(i + 1) * batch_size]
            output[i * batch_size:(i + 1) * batch_size] = extractor(batch_X).cpu().numpy()

    return output


def get_layer_wise_activations(model, dataset):
    """
    获取深度激活输出。
    :param model: PyTorch 模型。
    :param dataset: 数据集类型，'mnist'、'cifar' 或 'svhn'，不同数据集的架构可能不同。
    :return: 每层激活的列表。
    """
    assert dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset 参数必须是 'mnist', 'cifar' 或 'svhn'"

    # 遍历模型的所有层，收集输入和输出
    acts = []
    acts.append("Input")  # 输入层标识（可以替换为实际输入数据的形状）

    # 遍历模型的所有子模块，记录输出
    for layer in model.children():
        acts.append(layer)

    return acts

# lid of a single query point x
def mle_single(data, x, k=20):
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    # print('x.ndim',x.ndim)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    # dim = x.shape[1]

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]

# lid of a batch of query points X
def mle_batch(base, neighbors, k):
    """
    Maximum Likelihood Estimation (MLE) of local intrinsic dimensionality.
    :param base: Base points.
    :param neighbors: Neighbor points.
    :param k: Number of nearest neighbors.
    :return: LID estimates.
    """
    # Compute pairwise distances
    distances = np.linalg.norm(base[:, None] - neighbors[None, :], axis=2)
    distances = np.sort(distances, axis=1)[:, :k]
    # MLE for LID
    r = distances[:, -1] / distances[:, :-1]
    return -1 / np.mean(np.log(r + 1e-10), axis=1)


def get_lids_random_batch(model, X, X_noisy, X_adv, device, k=10, batch_size=100):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbors in the random batch it lies in.
    :param model: PyTorch model.
    :param X: Normal images.
    :param X_noisy: Noisy images.
    :param X_adv: Adversarial images.
    :param device: Device ('cpu' or 'cuda').
    :param k: Number of nearest neighbors for LID estimation.
    :param batch_size: Batch size (default 100).
    :return: lids, lids_noisy, lids_adv.
    """
    model = model.to(device).eval()
    lid_dim = len(list(model.children()))  # Number of layers
    print("Number of layers to estimate: ", lid_dim)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = min(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start

        # Select batch data
        X_batch = X[start:end].to(device)
        X_noisy_batch = X_noisy[start:end].to(device)
        X_adv_batch = X_adv[start:end].to(device)

        lid_batch = np.zeros((n_feed, lid_dim))
        lid_batch_adv = np.zeros((n_feed, lid_dim))
        lid_batch_noisy = np.zeros((n_feed, lid_dim))

        # Compute activations for all layers
        acts = get_layer_wise_activations(model, X_batch, device)
        acts_noisy = get_layer_wise_activations(model, X_noisy_batch, device)
        acts_adv = get_layer_wise_activations(model, X_adv_batch, device)

        # Estimate LID for each layer
        for i in range(lid_dim):
            lid_batch[:, i] = mle_batch(acts[i], acts[i], k=k)
            lid_batch_adv[:, i] = mle_batch(acts[i], acts_adv[i], k=k)
            lid_batch_noisy[:, i] = mle_batch(acts[i], acts_noisy[i], k=k)

        return lid_batch, lid_batch_noisy, lid_batch_adv

    lids, lids_noisy, lids_adv = [], [], []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_noisy, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)
        lids_noisy.extend(lid_batch_noisy)

    return (
        np.asarray(lids, dtype=np.float32),
        np.asarray(lids_noisy, dtype=np.float32),
        np.asarray(lids_adv, dtype=np.float32),
    )

    lids = []
    lids_adv = []
    lids_noisy = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_noisy, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)
        lids_noisy.extend(lid_batch_noisy)
        # print("lids: ", lids.shape)
        # print("lids_adv: ", lids_noisy.shape)
        # print("lids_noisy: ", lids_noisy.shape)

    lids = np.asarray(lids, dtype=np.float32)
    lids_noisy = np.asarray(lids_noisy, dtype=np.float32)
    lids_adv = np.asarray(lids_adv, dtype=np.float32)

    return lids, lids_noisy, lids_adv





# Mean distance of x to its k nearest neighbors
def kmean_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)
    k = min(k, len(data) - 1)
    f = lambda v: np.mean(v)
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

# Mean distance of x to its k nearest neighbors with PCA
def kmean_pca_batch(data, batch, k=10):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)
    a = np.zeros(batch.shape[0])
    for i in np.arange(batch.shape[0]):
        tmp = np.concatenate((data, [batch[i]]))
        tmp_pca = PCA(n_components=2).fit_transform(tmp)
        a[i] = kmean_batch(tmp_pca[:-1], tmp_pca[-1], k=k)
    return a



def get_kmeans_random_batch(model, X, X_noisy, X_adv, device, k=10, batch_size=100, pca=False):
    """
    Get the mean distance of each Xi in X_adv to its k nearest neighbors.

    :param model: PyTorch model.
    :param X: Normal images.
    :param X_noisy: Noisy images.
    :param X_adv: Adversarial images.
    :param device: Device ('cpu' or 'cuda').
    :param k: Number of nearest neighbors for K-means estimation.
    :param batch_size: Batch size (default 100).
    :param pca: Whether to apply PCA for distance calculation.
    :return: kms_normal: K-means of normal images (num_examples, 1)
            kms_noisy: K-means of noisy images (num_examples, 1)
            kms_adv: K-means of adversarial images (num_examples, 1)
    """
    model = model.to(device).eval()
    km_dim = len(list(model.children()))  # Number of layers
    print("Number of layers to use: ", km_dim)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = min(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start

        # Select batch data
        X_batch = X[start:end].to(device)
        X_noisy_batch = X_noisy[start:end].to(device)
        X_adv_batch = X_adv[start:end].to(device)

        km_batch = np.zeros((n_feed, km_dim))
        km_batch_adv = np.zeros((n_feed, km_dim))
        km_batch_noisy = np.zeros((n_feed, km_dim))

        # Compute activations for all layers
        acts = get_layer_wise_activations(model, X_batch, device)
        acts_noisy = get_layer_wise_activations(model, X_noisy_batch, device)
        acts_adv = get_layer_wise_activations(model, X_adv_batch, device)

        # Estimate K-means for each layer
        for i in range(km_dim):
            if pca:
                km_batch[:, i] = kmean_pca_batch(acts[i], acts[i], k=k)
            else:
                km_batch[:, i] = kmean_batch(acts[i], acts[i], k=k)
            if pca:
                km_batch_adv[:, i] = kmean_pca_batch(acts[i], acts_adv[i], k=k)
            else:
                km_batch_adv[:, i] = kmean_batch(acts[i], acts_adv[i], k=k)
            if pca:
                km_batch_noisy[:, i] = kmean_pca_batch(acts[i], acts_noisy[i], k=k)
            else:
                km_batch_noisy[:, i] = kmean_batch(acts[i], acts_noisy[i], k=k)

        return km_batch, km_batch_noisy, km_batch_adv

    kms, kms_adv, kms_noisy = [], [], []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        km_batch, km_batch_noisy, km_batch_adv = estimate(i_batch)
        kms.extend(km_batch)
        kms_adv.extend(km_batch_adv)
        kms_noisy.extend(km_batch_noisy)
    kms = np.asarray(kms, dtype=np.float32)
    kms_noisy=  np.asarray(kms_noisy, dtype=np.float32)
    kms_adv = np.asarray(kms_noisy, dtype=np.float32)
    return kms, kms_noisy, kms_adv

def score_point(tup):
    """
    TODO
    :param tup:
    :return:
    """
    x, kde = tup

    return kde.score_samples(np.reshape(x, (1, -1)))[0]


def score_samples(kdes, samples, preds, n_jobs=None):
    """
    TODO
    :param kdes:
    :param samples:
    :param preds:
    :param n_jobs:
    :return:
    """
    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()
    results = np.asarray(
        p.map(
            score_point,
            [(x, kdes[i]) for x, i in zip(samples, preds)]
        )
    )
    p.close()
    p.join()

    return results


def normalize(normal, adv, noisy):
    """Z-score normalisation
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]


def train_lr(X, y):
    """
    TODO
    :param X: the data samples
    :param y: the labels
    :return:
    """
    lr = LogisticRegressionCV(n_jobs=-1).fit(X, y)
    return lr


def train_lr_rfeinman(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr


def compute_roc(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score


def compute_roc_rfeinman(probs_neg, probs_pos, plot=False):
    """
    TODO
    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

def random_split(X, Y):
    """
    Random split the data into 80% for training and 20% for testing
    :param X: 
    :param Y: 
    :return: 
    """
    print("random split 80%, 20% for training and testing")
    num_samples = X.shape[0]
    num_train = int(num_samples * 0.8)
    rand_pert = np.random.permutation(num_samples)
    X = X[rand_pert]
    Y = Y[rand_pert]
    X_train, X_test = X[:num_train], X[num_train:]
    Y_train, Y_test = Y[:num_train], Y[num_train:]

    return X_train, Y_train, X_test, Y_test

def block_split(X, Y):
    """
    Split the data into 80% for training and 20% for testing
    in a block size of 100.
    :param X: 
    :param Y: 
    :return: 
    """
    print("Isolated split 80%, 20% for training and testing")
    num_samples = X.shape[0]
    partition = int(num_samples / 3)
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: 2*partition], Y[partition: 2*partition]
    X_noisy, Y_noisy = X[2*partition:], Y[2*partition:]
    num_train = int(partition*0.008) * 100

    X_train = np.concatenate((X_norm[:num_train], X_noisy[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_noisy[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_noisy[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_noisy[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    # unit test
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    c = np.array([11, 12, 13, 14, 15])

    a_z, b_z, c_z = normalize(a, b, c)
    print(a_z)
    print(b_z)
    print(c_z)
