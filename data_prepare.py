import collections
import torchtext
import os
import random
import torch
from torchtext.vocab import vocab, GloVe
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import device

torch.manual_seed(1234)
BATCH_SIZE = 64
device = device("cuda" if torch.cuda.is_available() else "cpu")


class ImdbDataset(Dataset):
    def __init__(
        self, folder_path="../dataset/aclImdb", is_train=True, is_small=False
    ) -> None:
        super().__init__()
        self.data, self.labels = self.read_dataset(folder_path, is_train, is_small)

    # 读取数据
    def read_dataset(
        self,
        folder_path,
        is_train,
        is_small,
    ):
        data, labels = [], []
        for label in ("pos", "neg"):
            folder_name = os.path.join(
                folder_path, "train" if is_train else "test", label
            )
            for file in tqdm(os.listdir(folder_name)):
                with open(os.path.join(folder_name, file), "rb") as f:
                    text = f.read().decode("utf-8").replace("\n", "").lower()
                    data.append(text)
                    labels.append(1 if label == "pos" else 0)
        # random.shuffle(data)
        # random.shuffle(labels)
        # 小样本训练，仅用于本机验证
        if is_small:
            data = data[:100]
            labels = labels[:100]
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], int(self.labels[index])

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels


def get_tokenized(data):
    """获取数据集的词元列表"""

    def tokenizer(text):
        return [tok.lower() for tok in text.split(" ")]

    return [tokenizer(review) for review in data]


def get_vocab(data):
    """获取数据集的词汇表"""
    tokenized_data = get_tokenized(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # 将min_freq设置为5，确保仅包括至少出现5次的单词
    vocab_freq = {"<UNK>": 0, "<PAD>": 1}
    # 添加满足词频条件的单词到词汇表，并分配索引
    for word, freq in counter.items():
        if freq >= 5:
            vocab_freq[word] = len(vocab_freq)

    # 构建词汇表对象并返回
    return vocab(vocab_freq)


def preprocess_imdb(train_data, vocab):
    """数据预处理，将数据转换成神经网络的输入形式"""
    max_l = 800  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [1] * (max_l - len(x))

    labels = train_data.get_labels()
    tokenized_data = get_tokenized(train_data.get_data())
    vocab_dict = vocab.get_stoi()
    features = torch.tensor(
        [pad([vocab_dict.get(word, 0) for word in words]) for words in tokenized_data]
    )
    labels = torch.tensor([label for label in labels])
    return features, labels


def load_data(batch_size):
    """加载数据集"""
    train_data = ImdbDataset(folder_path="./data/aclImdb", is_train=True)
    test_data = ImdbDataset(folder_path="./data/aclImdb", is_train=False)
    vocab = get_vocab(train_data.get_data())
    train_set = TensorDataset(*preprocess_imdb(train_data, vocab))
    # 20%作为验证集
    # train_set, valid_set = torch.utils.data.random_split(
    #     train_set, [int(len(train_set) * 0.8), int(len(train_set) * 0.2)]
    # )
    test_set = TensorDataset(*preprocess_imdb(test_data, vocab))
    train_iter = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    # valid_iter = DataLoader(valid_set, batch_size)
    test_iter = DataLoader(test_set, batch_size)
    return train_iter, test_iter, vocab


if __name__ == "__main__":
    train_data = ImdbDataset(is_train=True, is_small=True)
    test_data = ImdbDataset(is_train=False, is_small=True)
    vocab = get_vocab(train_data.get_data())
    print(f"词表中单词个数:{len(vocab)}")
    train_set = TensorDataset(*preprocess_imdb(train_data, vocab))
    test_set = TensorDataset(*preprocess_imdb(test_data, vocab))
    train_dataloader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE)
