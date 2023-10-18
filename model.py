from data_prepare import *
import torch
from torch import nn

torch.random.seed()


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(
            embed_size, num_hiddens, num_layers=num_layers, bidirectional=True
        )
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，
        # 所以在获得词元表示之前，输入会被转置。
        # 输出形状为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 返回上一个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


# 本函数已保存在d2lzh_torch包中方便以后使用
def load_pretrained_embedding(words, pretrained_vocab):
    """从预训练好的vocab中提取出words对应的词向量"""
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])  # 初始化为0
    oov_count = 0  # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 0
    if oov_count > 0:
        print("There are %d oov words.")
    return embed


if __name__ == "__main__":
    DATA_ROOT = "../dataset"
    embed_size, num_hiddens, num_layers = 100, 100, 2
    train_data = ImdbDataset(is_train=True, is_small=True)
    test_data = ImdbDataset(is_train=False, is_small=True)
    vocab = get_vocab(train_data.get_data())
    net = LSTM(len(vocab), embed_size, num_hiddens, num_layers)
    glove_vocab = GloVe(name="6B", dim=100, cache=os.path.join(DATA_ROOT, "glove"))
    net.embedding.weight.data.copy_(
        load_pretrained_embedding(vocab.get_itos(), glove_vocab)
    )
    net.embedding.weight.requires_grad = False  # 直接加载预训练好的, 所以不需要更新它

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(init_weights)
    print("模型初始化完成")
