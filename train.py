import logging
import time
from data_prepare import *
from model import *
import torch
from torch import nn
from torch import device
import os


device = device("cuda" if torch.cuda.is_available() else "cpu")


# 模型训练
def train_model():
    DATA_ROOT = "../dataset"
    embed_size, num_hiddens, num_layers = 100, 100, 2
    batch_size = 64
    train_iter, test_iter, vocab = load_data(batch_size)
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
    lr, num_epoch = 0.01, 5
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=lr
    )
    loss = nn.CrossEntropyLoss()

    print("--------------开始训练--------------")
    # 编写训练代码
    train(train_iter, test_iter, net, loss, optimizer, num_epoch)


def train(train_iter, test_iter, model, loss, optimizer, num_epoch):
    epoch_acces = []
    epoch_losses = []
    for epoch in range(num_epoch):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        start_time = time.time()
        for batch in train_iter:
            feature, label = batch
            feature, label = feature.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(feature)
            l = loss(output, label)
            l.backward()
            optimizer.step()
            epoch_loss += l.item()
            epoch_acc += (output.argmax(1) == label).sum().item()
        avg_train_loss = epoch_loss / len(train_iter)
        avg_train_acc = epoch_acc / len(train_iter)
        end_time = time.time()
        train_time = end_time - start_time
        print(
            f"=== epoch: {epoch+1}/{num_epoch}, train loss: {avg_train_loss}, train acc: {avg_train_acc}, train time: {train_time}s"
        )

        model.eval()
        eval_loss = 0
        eval_acc = 0
        for batch in test_iter:
            feature, label = batch
            feature, label = feature.to(device), label.to(device)
            output = model(feature)
            l = loss(output, label)
            eval_loss += l.item()
            eval_acc += (output.argmax(1) == label).sum().item()
        avg_eval_loss = eval_loss / len(test_iter)
        avg_eval_acc = eval_acc / len(test_iter)
        epoch_losses.append(avg_eval_loss)
        epoch_acces.append(avg_eval_acc)
    print("验证集上的损失为：", min(epoch_losses))
    print("验证集上的准确率为：", max(epoch_acces))
    print("-------------训练结束---------------")


if __name__ == "__main__":
    train_model()
