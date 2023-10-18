import logging
import time
from data_prepare import *
from model import *
import torch
from torch import nn
from torch import device
import os


device = device("cuda" if torch.cuda.is_available() else "cpu")
LR, NUM_EPOCH = 0.005, 10
BATCH_SIZE = 128
DATA_ROOT = "../dataset"


# 模型训练
def train_model():
    embed_size, num_hiddens, num_layers = 100, 100, 2
    batch_size = BATCH_SIZE
    train_iter, test_iter, vocab = load_data(batch_size)
    if os.path.exists("model.pt"):
        net = torch.load("model.pt")
        print("模型加载完成")
    else:
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

    lr, num_epoch = LR, NUM_EPOCH
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=lr
    )
    loss = nn.CrossEntropyLoss()

    train(train_iter, test_iter, net, loss, optimizer, num_epoch)


def train(train_iter, test_iter, model, loss, optimizer, num_epoch):
    print("===============开始训练===============")
    model.to(device)
    epoch_losses = []
    epoch_acces = []
    batch_count = 0
    start_train_time = time.time()
    for epoch in range(num_epoch):
        epoch_loss, epoch_acc, word_count, start_time = 0.0, 0.0, 0, time.time()
        model.train()
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
            batch_count += 1
            word_count += label.shape[0]
        avg_train_loss = epoch_loss / batch_count
        avg_train_acc = epoch_acc / word_count
        print(
            f"=== epoch: {epoch+1}/{num_epoch}, train loss: {avg_train_loss}, train acc: {avg_train_acc}, train time: {time.time()-start_time}s ==="
        )

        model.eval()
        eval_loss, eval_acc, n = 0.0, 0.0, 0
        for batch in test_iter:
            feature, label = batch
            feature, label = feature.to(device), label.to(device)
            output = model(feature)
            l = loss(output, label)
            eval_loss += l.item()
            eval_acc += (output.argmax(1) == label).sum().item()
            n += label.shape[0]
        avg_eval_loss = eval_loss / len(test_iter)
        avg_eval_acc = eval_acc / n
        epoch_losses.append(avg_eval_loss)
        epoch_acces.append(avg_eval_acc)
        # 每10个epoch，对结果进行输出
        if (epoch + 1) % 5 == 0:
            print(
                f"=== epoch: {epoch+1}/{num_epoch}, eval loss: {avg_eval_loss}, eval acc: {avg_eval_acc} ==="
            )
    print("训练总耗时：", time.time() - start_train_time)
    print("loss on valid set:", min(epoch_losses))
    print("acc on valid set:", max(epoch_acces))
    print("==============训练结束==============")
    torch.save(model, "model.pt")


def test():
    """测试集结果验证"""
    pass



