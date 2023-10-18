from train import *
from data_prepare import *
from model import *


def predict(model, vocab, sentence):
    """预测句子的情感"""
    device = list(model.parameters())[0].device
    vocab_dict = vocab.get_stoi()
    sentence = torch.tensor([vocab_dict[word] for word in sentence], device=device)
    label = torch.argmax(model(sentence.view((1, -1))), dim=1)
    return "positive" if label== 1 else "negative"


if __name__ == "__main__":
    # train_model()
    model = torch.load("model.pt")
    train_data = ImdbDataset(folder_path="./data/aclImdb", is_train=True)
    vocab = get_vocab(train_data.get_data())
    sentence = ["this", "movie", "is", "so", "great"]
    result = predict(model, vocab, sentence)
    assert result == "positive"
