import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler

from mydataset import ImageDataLoader_set
from torchvision import transforms as tfs
import pandas as pd
from itertools import chain

from test_dataset import ImageData_set
# from demo_densenet import dense_net
# from demo_densenet_adddropout import DenseNet
from demo_addsenet import DenseNet
from demo_restnet152 import ResNet152
import numpy as np
from Se_resnet import SEResNet

"""
cuda out of memory：是因为参数过多的原因嘛？
那训练的过程也是可以的啊，为什么测试的时候就报这个问题呢,内存太小了！！！
"""
batch_size = 10
num_epochs = 100
learning_rate = 0.001
train_image_path = "2increase_img224_224_resnet152"
label_path = "train_labels-increase.csv"
test_pacth = "test224_224for resnet152"
# 再看一下怎么归一化
data_tf = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
"""
# 添加验证集
dataset = ImageDataLoader_set(label_train_file=label_path, img_train_file=train_image_path)
# 就900吧
validation_split = 1 / 3
# shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
# print(indices)  # 返回一个18000的列表，然后开始切片
split = int(np.floor(validation_split * dataset_size))
# 要考虑的是切片打乱是就一次呢，还是每次进来都打乱一次，再考虑吧
# 不shuffle其实也可以
# 可能这里有点问题，因为我前面1800张其实是原始图像会不会对验证集和测试集有影响？
train_indices, val_indices = indices[split:], indices[:split]  # 0-split为验证集，split-18000为训练集
# 构造train和val
# SubsetRandomSampler 无放回的按照indices列表给值，但是有点shuffle的感觉，API没说这件事，只不过现在是不放回就行了
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# 构造dataloader
# 我透，sample不能与shuffle同时用;好像这个采样过程就是随机的，不知道会不会采到同一幅图片:不会
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
"""
# 没有验证集的data loader,数据量是9000
train_set = ImageDataLoader_set(label_train_file=label_path, img_train_file=train_image_path)
test_set = ImageData_set(img_test_file=test_pacth)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
# 把test 放到另一个py文件中吧 ，把最优的epoch写完，不用这一步了，没事
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
"""
1）
# dense_net分类,收敛比较快，需添加正则化---->得到了72.多的精度（在10epoch的时候）
demo_dense_net = dense_net(3, 3)
demo_dense_net.cuda()
"""
# 工作1：虽然loss上下跳动，但是收敛的速度还是有点明显，可以适当调整drop_rate（调高）去看下得到的结果（先不看，先做2和3以及4）
# 工作2：用最优的模型去跑测试集和验证集，而不是用最后一个，也看到每次最优的模型并不是最后一个
# 工作3：将训练集的10%用作验证集，这样就可以不用每次去提交了（已完成）
# 工作4：将数据集再次扩大，想想扩大的方式（已完成）
# 工作5：进行网络的调参工作
# 工作6：可以想想添加self_attention(先去了解一下)

"""
3）
在完成工作3，工作4后开始训练，并查看val_truth,[会发现随着epoch的增加，每一个epoch的loss_mean在减少，也有例外（反弹了），收敛的速度明显变慢了，可以降低lr]
epoch:20,lr:0.001,best epoch:19 ,loss_mean:0.263185,val:0.684444
epoch:20,lr:0.0001,best epoch:30 ,loss_mean:0.109046,val:0.751667
"""
"""
2）
添加了drop_out（正则项）的dense_net，

1）epoch:25,learning_rate=0.001,best epoch:24 ,loss_mean:0.139724 score:0.7127
2) epoch:30,learning_rate=0.0001,best epoch:29 ,loss_mean:0.102249 score:不看因为以上没有用到最优的epoch
"""
# demo_dense_net = DenseNet()
# demo_dense_net.cuda()

# se_resnet = SEResNet(3)
# se_resnet.cuda()
demo_resnet152 = ResNet152()
demo_resnet152.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(demo_resnet152.parameters(), lr=learning_rate)


# train
def train():
    for epoch in range(num_epochs):
        loss_all = 0
        best_loss_all = 10
        for idx, (images, labels) in enumerate(train_loader):
            images = Variable(images.to(torch.float)).cuda()  # batchsize *3* 128*173tensor
            # images = images.view(batch_size, -1, 128, 173)
            # b = images.data.cpu().numpy()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = demo_resnet152(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (idx + 1) % 90 == 0:
                loss_all += loss
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.6f'
                      % (epoch + 1, num_epochs, idx + 1, len(train_loader), loss.item()))
        loss_all_mean = loss_all / 10
        print('loss_all_mean:%.6f' % loss_all_mean)
        if loss_all_mean <= best_loss_all:
            best_loss_all = loss_all_mean
            best_loss_epoch = epoch + 1
            if best_loss_all < 0.125:
                print("best epoch:%d ,loss_mean:%.6f," % (best_loss_epoch, best_loss_all))
                test()
                # score = val()
                # if score > 0.855:
                #    test()
                # else:
                #    print("-----------------------------------------------")
                #   train()


"""
# val:6000张图像
def val():
    sum = 0
    error = 0
    # labels 原本就是cpu里面的
    for idx, (images, labels) in enumerate(validation_loader):
        images = Variable(images.to(torch.float)).cuda()
        # labels = Variable(labels).cuda()
        labels = labels.numpy()
        outputs = demo_resnet152(images)
        # loss = criterion(outputs, labels)  # batchsize的loss,不用这么写
        _, predicted = torch.max(outputs.data, 1)  # 计算验证集的label
        predicted_label = predicted.cpu().numpy()
        for i in range(len(predicted_label)):
            if predicted_label[i] != labels[i]:
                error += 1
                sum += 1
            else:
                sum += 1

    score = (sum - error) / sum
    print('val:%.6f' % score)
    return score
"""


# Test分出来到另一个py
# Test the Model
def test():
    test_accent = []
    for idx, (images) in enumerate(test_loader):
        images = Variable(images.to(torch.float)).cuda()
        outputs = demo_resnet152(images)
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


train()
