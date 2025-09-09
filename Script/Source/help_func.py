import os

import yaml

import random
import numpy as np
import torch


def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)

    return cfg


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def tensor2array(tensor, squeeze=False):
    """
    transfer the torch tensor to numpy array
    ----------------------------
    Parameters:
        tensor: [torch tensor] tensor to be transferred
        squeeze: [Bool] option to squeeze the tensor dimensionality
    Return:
        numpy array
    ----------------------------
    """
    if squeeze:
        tensor = tensor.squeeze()
    return tensor.detach().cpu().numpy()


def seed_torch(seed=0):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True
