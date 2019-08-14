import numpy as np
import cv2
import os
import random
import pandas as pd
from torch.utils.data import Dataset


class ImageDataLoader_set(Dataset):

    def __init__(self, label_train_file, img_train_file):
        self.train_labels = pd.read_csv(label_train_file)
        self.img_train_file = img_train_file
        self.img_names = self.load_data(self.img_train_file)

    def __getitem__(self, index):
        # 获取label
        label_id_accent = self.train_labels['accent'][index]
        img = cv2.imread(os.path.join(self.img_train_file, self.img_names[index]))  # 以3通道读取方式，128*173*3
        img = img.reshape(3, 224, 224)
        # img = cv2.resize(img, (224, 224))
        return img, label_id_accent

    def __len__(self):
        return len(self.train_labels)

    def load_data(self, path):
        image_names = [filename for filename in os.listdir(path) \
                       if os.path.isfile(os.path.join(path, filename))]
        return image_names
