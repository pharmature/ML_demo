# RNNcell
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

cell = torch.nn.RNNCell(input_size = input_size,hidden_size=hidden_size)

#(seq,batch,features)
dataset = torch.randn(seq_len,batch_size,input_size)
hidden = torch.zeros(batch_size,hidden_size)

for idx,input in enumerate(dataset):
    print('='*20,idx,'='*20)
    print('Input size',input.shape)

    hidden = cell(input,hidden)

    print('output size',hidden.shape)
    print(hidden)


# RNN
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
#比rnncell多一个numlayers
num_layers = 1

cell = torch.nn.RNN(input_size = input_size,hidden_size=hidden_size,
                        num_layers = num_layers)

#(seq,batch,features)
# 指明维度
inputs = torch.randn(seq_len,batch_size,input_size)
hidden = torch.zeros(num_layers,batch_size,hidden_size)

out,hidden = cell(inputs,hidden)

print('Ouput size',out.shape)
print('Output:',out)
print('gidden size',hidden.shape)
print('hidden:',hidden)


# hello -> ohlol
import torch
from torch import nn, optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

input_size = 4
hidden_size = 4
num_layers = 1
batch_size = 1
seq_len = 5
# 准备数据
idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]  # 分别对应0,1,2,3项
x_one_hot = [one_hot_lookup[x] for x in x_data]  # 组成序列张量
print('x_one_hot:', x_one_hot)

# 构造输入序列和标签
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)  # labels维度是: (seqLen * batch_size ，1)


# design model
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        # 为了能和labels做交叉熵，需要reshape一下:(seqlen*batchsize, hidden_size),即二维向量，变成一个矩阵
        return out.view(-1, self.hidden_size)


net = Model(input_size, hidden_size, batch_size, num_layers)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

# train cycle
epoch_list = []
loss_list = []
for epoch in range(20):
    optimizer.zero_grad()
    # inputs维度是: (seqLen, batch_size, input_size) labels维度是: (seqLen * batch_size * 1)
    # outputs维度是: (seqLen, batch_size, hidden_size)
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(',Epoch [%d/20] loss=%.3f' % (epoch + 1, loss.item()))
    epoch_list.append(epoch)
    loss_list.append(loss.item())


plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
