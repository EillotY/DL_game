import numpy as np
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler

from mydataset import ImageDataLoader_set
import torch

my_path = 'train_4increase_all'
label_train_path = 'train_labels-4increase.csv'
dataset = ImageDataLoader_set(label_train_file=label_train_path, img_train_file=my_path)
batch_size = 25
# 就1800吧
validation_split = .1
# shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
# print(indices)  # 返回一个18000的列表，然后开始切片
split = int(np.floor(validation_split * dataset_size))
# 要考虑的是切片打乱是就一次呢，还是每次进来都打乱一次，再考虑吧
# 不shuffle其实也可以，虽然验证集每次都是一样，但是我的训练集每个epoch都是shuffle，并且不知道验证集的状态的
"""
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
"""
train_indices, val_indices = indices[split:], indices[:split]  # 0-split为验证集，split-18000为训练集
print(train_indices)
print(val_indices)
# 构造train和val
# SubsetRandomSampler 无放回的按照indices列表给值，但是有点shuffle的感觉，API没说这件事，只不过现在是不放回就行了
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# 构造dataloader
# 我透，sample不能与shuffle同时用;好像这个采样过程就是随机的，不知道会不会采到同一幅图片
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

for idx, (images, labels) in enumerate(train_loader):
    images = Variable(images.to(torch.float)).cuda()  # batchsize *3* 128*173tensor
    # images = images.view(batch_size, -1, 128, 173)
    # b = images.data.cpu().numpy()
    labels = Variable(labels).cuda()