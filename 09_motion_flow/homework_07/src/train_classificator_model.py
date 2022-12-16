from train_data_generator import (
    SAVE_POSITIVE_PATH,
    SAVE_NEGATIVE_PATH
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

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    # to normalize correctly one should calculated mean and std
    # other whole dataset
    # transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    transforms.Lambda(torch.flatten)
])


def train_image_preprocessing(image: cv2.Mat) -> torch.Tensor:
    image = TRANSFORM(image)
    return image


class SimplePerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1),
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
    def __init__(self):

        self.images = []
        self.labels = []

        for img_name in tqdm(os.listdir(SAVE_POSITIVE_PATH)):
            self.images += [
                train_image_preprocessing(
                    cv2.imread(SAVE_POSITIVE_PATH + img_name)
                )
            ]
            self.labels += [1]

        for img_name in tqdm(os.listdir(SAVE_NEGATIVE_PATH)):
            self.images += [
                train_image_preprocessing(
                    cv2.imread(SAVE_NEGATIVE_PATH + img_name)
                )
            ]
            self.labels += [0]

        self.labels = np.array(self.labels, dtype=np.float32)[..., np.newaxis]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


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
    running_loss = 0.0
    processed_size = 0

    with tqdm(total=len(val_loader)) as pbar:
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            pbar.update(1)
            running_loss += loss.item() * inputs.size(0)
            processed_size += inputs.size(0)

    val_loss = float(running_loss / processed_size)
    return val_loss


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
    val_loss = None

    for epoch in range(epochs):
        print(f"EPOCH {epoch}/{epochs}: ")
        train_loss = fit_epoch(model, train_loader, criterion, optimizer)
        if val_files is not None:
            val_loss = eval_epoch(model, val_loader, criterion)

        if val_files is not None:
            print("Train loss: ", train_loss)
            print("Validation loss: ", val_loss)
            history.append((train_loss, val_loss))
        else:
            print("Train loss: ", train_loss)
            history.append((train_loss))

    return history


def plot_history(history: List) -> None:
    plt.figure(figsize=(5, 4))
    plt.plot(history, label="train_loss")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()


def train_model():
    model = SimplePerceptron().to(DEVICE)
    train_files = PlanesDataset()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5
    )
    criterion = torch.nn.BCELoss()

    history = train(
        train_files=train_files,
        val_files=None,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        epochs=20,
        batch_size=128
    )
    torch.save(model, MODEL_SAVE_PATH + "model")
    # print(history)
    # plot_history(history)


if __name__ == "__main__":
    train_model()
