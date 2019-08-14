import torch
import torch.nn as nn
from torch.autograd import Variable
from mydataset import ImageDataLoader_set
from torchvision import transforms as tfs
import pandas as pd
from itertools import chain

from test_dataset import ImageData_set

"""
@author:Fenta_Yuan
数据相当于shuffle的状态，如果得不到好的结果 得想下其他的方式；
1）可以添加validation
2) 选取网络中最优的epoch里的某一步step的参数，进行后面的test
3）想想用fine_tune
4) 要做数据增强,训练数据太少了
"""

"""
1）
结论：LSTM 不适合做图片分类 【2019/7/22之前的理解】
放弃，改为采用densenet 卷积实现
"""

input_size = 173
hidden_size = 1000
num_layers = 2
num_classes = 3
batch_size = 50
num_epochs = 1
learning_rate = 0.0001

train_image_path = "train-increase"
label_path = "train_labels-increase.csv"
test_pacth = "test"
data_tf = tfs.Compose([
    tfs.ToTensor(),
])

train_set = ImageDataLoader_set(label_train_file=label_path, img_train_file=train_image_path)
test_set = ImageData_set(img_test_file=test_pacth)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes):  # input_size=128，hidden_size=200，num_layers=2，num_classes=3
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # 173*200*2
        self.fc = nn.Linear(hidden_size, num_classes)  # 200 *3

    def forward(self, x):
        # h0 = torch.tensor(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(torch.float).cuda()
        # c0 = torch.tensor(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(torch.float).cuda()
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(torch.float)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(torch.float)).cuda()
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # 选择最后一个时间点的output
        out = self.fc(out[:, -1, :])

        return out


rnn = RNN(input_size, hidden_size, num_layers, num_classes)
rnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

# train
for epoch in range(num_epochs):
    loss_all = 0
    best_loss_all = 100
    best_loss_epoch = epoch
    for idx, (images, labels) in enumerate(train_loader):
        images = Variable(images.to(torch.float)).cuda()
        # b = images.data.cpu().numpy()
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        outputs = rnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (idx + 1) % 5 == 0:
            loss_all += loss
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.6f'
                  % (epoch + 1, num_epochs, idx + 1, len(train_loader), loss.item()))
    loss_all_mean = loss_all / (180 / 5)
    if loss_all_mean <= best_loss_all:
        best_loss_all = loss_all_mean
        best_loss_epoch = epoch
print("best epoch:%d ,loss_mean:%.6f," % (epoch, best_loss_all))
# Test the Model
test_accent = []
for idx, (images) in enumerate(test_loader):
    images = Variable(images.to(torch.float)).cuda()
    outputs = rnn(images)
    _, predicted = torch.max(outputs.data, 1)
    test_accent.append(predicted.cpu().numpy())  # 首先predicted是Gpu tensor 我先把它转为cpu tensor
# print(test_accent)
a = list(chain.from_iterable(test_accent))
testcsv = pd.read_csv('test_labels.csv')
for idx in range(len(a)):
    testcsv['accent'][idx] = a[idx]
print(testcsv)
save = pd.DataFrame(testcsv)
save.to_csv('test_labels.csv', index=False, header=True)
