import numpy as np
import cv2
import os
import random
import pandas as pd
from torch.utils.data import Dataset


class ImageData_set(Dataset):

    def __init__(self, img_test_file):
        self.img_test_file = img_test_file
        self.img_names = self.load_data(self.img_test_file)

    def __getitem__(self, index):
        # 获取label
        img = cv2.imread(os.path.join(self.img_test_file, self.img_names[index]))
        # img = img.reshape(3, 128, 173)
        img = img.reshape(3, 224, 224)
        return img

    def __len__(self):
        return len(self.img_names)

    def load_data(self, path):
        image_names = [filename for filename in os.listdir(path) \
                       if os.path.isfile(os.path.join(path, filename))]
        return image_names
