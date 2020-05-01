import numpy as np
import os
from random import Random
import glob
import sys
import imageio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

"""
This code is inspired from 
HW2 of https://cs330.stanford.edu/
"""

def get_images(paths, labels, random, nb_samples=None, shuffle=True):
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
    images_labels = [
        (i, os.path.join(path, image))
        for i, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
    ]
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
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class Datagenerator(Dataset):
    def __init__(
        self, num_classes, num_samples_per_class, data_folder, img_size, dataset_type
    ):
        """
        Args:
            num_classes: Number of classes for classification
            num_samples_per_class: num samples per class
            data_folder: Data folder
            image_size: Image size
        """
        self.num_classes = num_classes
        # Multiplied by 2 to get outer and inner inputs
        self.num_samples_per_class = 2 * num_samples_per_class
        self.dim_input = np.prod(img_size)
        self.dim_output = self.num_classes
        self.dataset_type = dataset_type
        self.random = Random(1)
        character_folders = sorted([
            os.path.join(data_folder, family, character)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for character in os.listdir(os.path.join(data_folder, family))
            if os.path.isdir(os.path.join(data_folder, family, character))
        ])
        np.random.seed(111)
        self.random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        if dataset_type == "train":
            self.character_folders = character_folders[:num_train]
        elif dataset_type == "val":
            self.character_folders = character_folders[num_train : num_train + num_val]
        elif dataset_type == "test":
            self.character_folders = character_folders[num_train + num_val :]
        else:
            raise ("Wrong dataset type: valid types are train, test and val")
        self.image_cache = self.load_images(self.character_folders, self.dim_input)

    def __getitem__(self, index):
        sampled_character_folders = self.random.sample(
            self.character_folders, self.num_classes
        )
        labels_and_images = get_images(
            sampled_character_folders,
            range(self.num_classes),
            random=self.random,
            nb_samples=self.num_samples_per_class,
            shuffle=False,
        )
        labels = [li[0] for li in labels_and_images]
        images = [self.image_cache[li[1]] for li in labels_and_images]
        ims = np.stack(images)
        labels = np.reshape(labels, (self.num_classes, self.num_samples_per_class))
        # labels shape = (num_classes, num_samples_per_class)
        ims = np.reshape(ims, (self.num_classes, self.num_samples_per_class, -1))
        # ims shape = (num_classes, num_samples_per_class, dim_input)
        inner_inputs, outer_inputs = (
            ims[:, 0 : self.num_samples_per_class // 2, :],
            ims[:, self.num_samples_per_class // 2 :, :],
        )
        inner_labels, outer_labels = (
            labels[:, 0 : self.num_samples_per_class // 2],
            labels[:, self.num_samples_per_class // 2 :],
        )
        # Shuffle the order of classes in both inner and outer inputs, so that the model does not memorize the order
        perm_inner = np.random.permutation(self.num_classes)
        perm_outer = np.random.permutation(self.num_classes)
        inner_inputs = inner_inputs[perm_inner, :]
        inner_labels = inner_labels[perm_inner, :]
        outer_inputs = outer_inputs[perm_outer, :]
        outer_labels = outer_labels[perm_outer, :]
        return {
            "inner_inputs": torch.FloatTensor(inner_inputs),
            "inner_labels": torch.LongTensor(inner_labels),
            "outer_inputs": torch.FloatTensor(outer_inputs),
            "outer_labels": torch.LongTensor(outer_labels),
        }

    def __len__(self):
        return int(1e6)

    def load_images(self, folders, dim_input):
        images = dict()
        for folder in tqdm(folders):
            files = glob.glob(folder + "/**/*.png", recursive=True)
            for f in files:
                images[f] = image_file_to_array(f, dim_input)
        return images
