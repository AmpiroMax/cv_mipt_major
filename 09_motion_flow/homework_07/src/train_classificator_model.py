from train_data_generator import (
    SAVE_TRAIN_POSITIVE_PATH,
    SAVE_TRAIN_NEGATIVE_PATH,
    SAVE_TEST_POSITIVE_PATH,
    SAVE_TEST_NEGATIVE_PATH,
    SEGMENT_SIZE
)

import os
import cv2
import torch
import numpy as np
import matplotlib as plt
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import List
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "09_motion_flow/homework_07/train_test_data/models/"
DATASET_STATISTICS = "09_motion_flow/homework_07/train_test_data/"

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([131.9550369378745], [45.437329084087004]),
])

AUGMENTATION = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    # transforms.RandomCrop(SEGMENT_SIZE // 2),
])


def dataset_image_preprocessing(image: cv2.Mat) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = TRANSFORM(image)
    return image


class SimplePerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        dim = SEGMENT_SIZE**2
        self.model = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim // 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim // 2),
            nn.Linear(dim // 2, dim // 4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim // 4),
            nn.Linear(dim // 4, dim // 8),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim // 8),
            nn.Linear(dim // 8, 1),
            nn.Sigmoid()
        )

        self.model.apply(self._init_weights)

    def forward(self, x):
        x = self.model(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class PlanesDataset(Dataset):
    def __init__(self, mode="train"):
        self.images = []
        self.labels = []
        self.mode = mode

        save_neg_path = SAVE_TRAIN_NEGATIVE_PATH
        save_pos_path = SAVE_TRAIN_POSITIVE_PATH
        if mode == "test":
            save_neg_path = SAVE_TEST_NEGATIVE_PATH
            save_pos_path = SAVE_TEST_POSITIVE_PATH

        for img_name in tqdm(os.listdir(save_pos_path)):
            self.images += [
                dataset_image_preprocessing(
                    cv2.imread(save_pos_path + img_name)
                )
            ]
            self.labels += [1]

        for img_name in tqdm(os.listdir(save_neg_path)):
            self.images += [
                dataset_image_preprocessing(
                    cv2.imread(save_neg_path + img_name)
                )
            ]
            self.labels += [0]

        self.labels = np.array(self.labels, dtype=np.float32)[..., np.newaxis]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]
        if self.mode == "train":
            image = AUGMENTATION(image)
        image = torch.flatten(image)
        return image, label


def fit_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.BCELoss,
    optimizer: torch.optim.Optimizer
) -> float:

    model.train()
    running_accuracy = 0.0
    processed_data = 0

    with tqdm(total=len(train_loader)) as pbar:
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            pbar.update(1)

            outputs[outputs < 0.5] = 0
            outputs[outputs >= 0.5] = 1
            running_accuracy += torch.sum(outputs == labels)
            processed_data += inputs.size(0)

    train_accuracy = float(running_accuracy / processed_data)
    return train_accuracy


def eval_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.BCELoss,
) -> float:

    model.eval()
    running_accuracy = 0.0
    processed_data = 0

    with tqdm(total=len(val_loader)) as pbar:
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                outputs[outputs < 0.5] = 0
                outputs[outputs >= 0.5] = 1
                running_accuracy += torch.sum(outputs == labels)
                processed_data += inputs.size(0)
            pbar.update(1)

    valid_accuracy = float(running_accuracy / processed_data)
    return valid_accuracy


def train(
    train_files: Dataset,
    val_files: Dataset,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.BCELoss,
    epochs: int,
    batch_size: int
):
    train_loader = DataLoader(
        train_files,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = None
    if val_files is not None:
        val_loader = DataLoader(
            val_files,
            batch_size=batch_size,
            shuffle=False
        )

    history = []
    train_loss = None
    val_accuracy = None
    best_model = None
    best_accuracy = 0.0

    for epoch in range(epochs):
        print(f"EPOCH {epoch}/{epochs}: ")
        train_loss = fit_epoch(model, train_loader, criterion, optimizer)
        if val_files is not None:
            val_accuracy = eval_epoch(model, val_loader, criterion)

        if val_files is not None:
            print("Train accuracy:      ", train_loss)
            print("Validation accuracy: ", val_accuracy)
            history.append((train_loss, val_accuracy))

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model

        else:
            print("Train accuracy: ", train_loss)
            history.append((train_loss))

    return history, best_model


def plot_history(history: List) -> None:
    plt.figure(figsize=(5, 4))
    plt.plot(history, label="train_loss")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()


def train_model():
    model = SimplePerceptron().to(DEVICE)
    train_files = PlanesDataset("train")
    val_files = PlanesDataset("test")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4
    )
    criterion = torch.nn.BCELoss()

    history, best_model = train(
        train_files=train_files,
        val_files=val_files,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        epochs=80,
        batch_size=1024
    )
    torch.save(best_model, MODEL_SAVE_PATH + "model")


if __name__ == "__main__":
    train_model()
