import torch.nn as nn
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from data.omniglot_dataset import Datagenerator
from inner_update import apply_inner_update, get_task_outer_loss
from tensor_logger import writer
from utils import dotdict, make_collate_fn
from collections import OrderedDict
from torchvision.utils import make_grid
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt

class MAML:
    def __init__(self, hparams, model):
        self.device = hparams.device
        self.model = model
        self.num_classes = hparams.num_classes
        self.num_samples_per_class = hparams.num_samples_per_class
        self.data_folder = hparams.data_folder
        self.num_meta_test_classes = hparams.num_meta_test_classes
        self.num_meta_test_samples_per_class = hparams.num_meta_test_samples_per_class
        self.meta_test_num_inner_updates = hparams.meta_test_num_inner_updates
        self.num_train_inner_updates = max(
            hparams.num_inner_updates, hparams.meta_test_num_inner_updates
        )
        self.train_iterations = hparams.num_meta_train_iterations
        self.validation_iterations = hparams.num_meta_validation_iterations
        self.test_iterations = hparams.num_meta_test_iterations
        self.batch_size = hparams.batch_size
        self.meta_lr = hparams.meta_lr
        self.inner_update_lr = hparams.inner_update_lr
        self.validation_frequency = hparams.validation_frequency
        self._train_data_loader = self.train_data_loader()
        self._test_data_loader = self.test_data_loader()
        self._validation_data_loader = self.validation_data_loader()
        self.dim_hidden = hparams.dim_hidden

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.test = self.build_evaluation_fn(
            self.test_iterations, self._test_data_loader, self.batch_size, "Test"
        )
        self.validate = self.build_evaluation_fn(
            self.validation_iterations,
            self._validation_data_loader,
            self.batch_size,
            "Validation",
        )


    def train_step(self):
        self.model.train()
        batch = next(iter(self._train_data_loader))
        mean_outer_loss = torch.tensor(0.0, device=self.device)
        accuracy = []
        for task_num in range(self.batch_size):
            self.train_batch_num += 1
            task_inner_inputs = batch["inner_inputs"][task_num]
            task_inner_labels = batch["inner_labels"][task_num]
            task_outer_inputs = batch["outer_inputs"][task_num]
            task_outer_labels = batch["outer_labels"][task_num]
            # Perform inner gradient descent for "num_inner_updates" steps

            writer.add_image(
                "train/inner", make_grid(task_inner_inputs), self.train_batch_num,
            )
            writer.add_image(
                "train/outer", make_grid(task_outer_inputs), self.train_batch_num,
            )
            outer_loss, outer_accuracy = get_task_outer_loss(
                self.model,
                self.loss_fn,
                task_inner_inputs,
                task_inner_labels,
                task_outer_inputs,
                task_outer_labels,
                self.inner_update_lr,
                self.num_train_inner_updates,
                "train",
            )

            mean_outer_loss += outer_loss
            accuracy.append(outer_accuracy)
        mean_outer_loss.div_(self.batch_size)

        self.optimizer.zero_grad()
        mean_outer_loss.backward()
        self.optimizer.step()

        mean_accuracy = np.mean(accuracy)
        writer.add_scalar(
            "train/mean_outer_loss", mean_outer_loss, self.train_batch_num
        )
        writer.add_scalar(
            "train/mean_outer_accuracy", mean_accuracy, self.train_batch_num
        )
        return mean_outer_loss, mean_accuracy

    def train(self):
        self.train_batch_num = 0
        with tqdm(total=self.train_iterations) as pbar:
            for itr in range(self.train_iterations):
                self.episode = itr

                mean_outer_train_loss, mean_outer_train_accuracy = self.train_step()
                pbar.update(1)
                postfix = {
                    "loss": f"{mean_outer_train_loss:.4f}",
                    "accuracy": f"{mean_outer_train_accuracy:.4f}",
                }
                pbar.set_description(f"Training itr = {itr+1}")
                pbar.set_postfix(**postfix)
                if itr % self.validation_frequency == 0:
                    self.validate()


    def build_evaluation_fn(self, num_iterations, data_loader, batch_size, prefix):
        def eval_fn():
            accuracies = []
            for itr in range(num_iterations):
                mean_outer_loss, batch_accuracies = self.evaluate(
                    data_loader, prefix
                )
                accuracies.append(batch_accuracies)
            accuracies = list(itertools.chain.from_iterable(accuracies))
            print(f"{prefix} Accuracy: {np.mean(accuracies)}")
            logfile_name = open(os.path.join("logs", f"{prefix}.csv"), "a")
            logfile_name.write(f"{self.episode},{np.mean(accuracies)}\n")
        return eval_fn

    def evaluate(self, data_iterator, prefix=""):
        self.model.eval()
        mean_outer_loss = torch.tensor(0.0, device=self.device)
        batch = next(iter(data_iterator))
        accuracies = []
        for task_num in range(self.batch_size):
            task_inner_inputs = batch["inner_inputs"][task_num]
            task_inner_labels = batch["inner_labels"][task_num]
            task_outer_inputs = batch["outer_inputs"][task_num]
            task_outer_labels = batch["outer_labels"][task_num]
            outer_loss, outer_accuracy = get_task_outer_loss(
                self.model,
                nn.CrossEntropyLoss(),
                task_inner_inputs,
                task_inner_labels,
                task_outer_inputs,
                task_outer_labels,
                self.inner_update_lr,
                self.num_train_inner_updates,
                prefix,
            )
            mean_outer_loss += outer_loss
            accuracies.append(outer_accuracy)
        mean_outer_loss.div_(self.batch_size)
        mean_accuracy = np.mean(accuracies)
        return mean_outer_loss, accuracies

    def train_data_loader(self):
        train_data_generator = Datagenerator(
            self.num_classes,
            self.num_samples_per_class,
            self.data_folder,
            (28, 28),  # Size of omniglot dataset images
            "train",
        )
        return DataLoader(
            train_data_generator,
            batch_size=self.batch_size,
            collate_fn=make_collate_fn(self.device),
        )

    def test_data_loader(self):
        test_data_generator = Datagenerator(
            self.num_meta_test_classes,
            self.num_meta_test_samples_per_class,
            self.data_folder,
            (28, 28),  # Size of omniglot dataset images
            "test",
        )
        return DataLoader(
            test_data_generator,
            batch_size=self.batch_size,
            collate_fn=make_collate_fn(self.device),
        )

    def validation_data_loader(self):
        val_data_generator = Datagenerator(
            self.num_classes,
            self.num_samples_per_class,
            self.data_folder,
            (28, 28),  # Size of omniglot dataset images
            "val",
        )
        return DataLoader(
            val_data_generator,
            batch_size=self.batch_size,
            collate_fn=make_collate_fn(self.device),
        )