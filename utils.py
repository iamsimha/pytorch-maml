import random
import os
import torch
import  numpy as np
from scipy import misc

def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                        for i, path in zip(labels, paths)
                        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    image = misc.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def set_module_prefix(model):
    """
    Set module prefix attributes for each of the modules, so that correct parameters can be extracted
    """
    memo = set()
    prefix = ""
    recurse = True
    get_members_fn = lambda module: module._parameters.items()
    modules = model.named_modules(prefix="") if recurse else [(prefix, model)]
    for module_prefix, module in modules:
        members = get_members_fn(module)            
        for k, v in members:
            if v is None or v in memo:
                continue
            memo.add(v)
            name = module_prefix + ('.' if module_prefix else '') + k
            module.module_prefix = module_prefix
            break

def make_collate_fn(device):
    def collate_fn(batch):
        """
        Passed to dataloader to collate items into a batch
        """
        nonlocal device
        inner_inputs = torch.stack([item["inner_inputs"] for item in batch])
        outer_inputs = torch.stack([item["outer_inputs"] for item in batch])
        inner_labels = torch.stack([item["inner_labels"] for item in batch])
        outer_labels = torch.stack([item["outer_labels"] for item in batch])
        batch_size = len(batch)
        return {
            "inner_inputs": inner_inputs.reshape(batch_size, -1, 1, 28, 28).to(device),
            "outer_inputs": outer_inputs.reshape(batch_size, -1, 1, 28, 28).to(device),
            "inner_labels": inner_labels.reshape(batch_size, -1).to(device),
            "outer_labels": outer_labels.reshape(batch_size, -1).to(device),
        }
    return collate_fn

def print_norm(model):
    for name, param in model.named_parameters():
        print(f"name = {name}, norm={torch.norm(param)}")

def norm_sum_data(model):
    res = 0
    for name, param in model.named_parameters():
        res += torch.norm(param.data)
    return res

def norm_sum_grad(model):
    res = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            res += torch.norm(param.grad)
    return res