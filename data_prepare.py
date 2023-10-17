import os
import numpy as np
import pandas as pd
import random
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

random.seed(2023)


class ImdbDataset(Dataset):
    def __init__(self, is_train=True) -> None:
        super().__init__()
        self.data, self.labels = self.read_dataset(is_train=is_train)

    # 读取数据
    def read_dataset(self, folder_path="../dataset/aclImdb", is_train=True):
        data, labels = [], []
        for label in ("pos", "neg"):
            folder_name = os.path.join(
                folder_path, "train" if is_train else "test", label
            )
            for file in os.listdir(folder_name):
                with open(os.path.join(folder_name, file), "rb") as f:
                    text = f.read().decode("utf-8").replace("\n", "")
                    data.append(text)
                    labels.append(1 if label == "pos" else 0)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], int(self.labels[index])


if __name__ == "__main__":
    train_dataset = ImdbDataset(is_train=True)
    test_dataset = ImdbDataset(is_train=False)
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=0
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=8, shuffle=True, num_workers=0
    )
    # print(len(train_dataset))
