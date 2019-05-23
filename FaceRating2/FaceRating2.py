import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from skimage.transform import resize


# load data
class Images(Dataset):

    def __init__(self, imagespath, labelspath, shape=(350, 350),
                 is_shuffle=True, mode='train'):

        self.img_shape = shape
        self.imagespath = sorted(glob.glob(os.path.join(imagespath, '*.*')))

        # 80% of the data as training data
        if mode == 'train':
            self.imagespath = self.imagespath[:int(len(self.imagespath) * 0.8)]
        elif mode == 'test':
            self.imagespath = self.imagespath[int(len(self.imagespath) * 0.8):]
        else:
            raise ValueError("mode should be 'train' or 'test', not %s" % mode)
        if is_shuffle:
            random.shuffle(self.imagespath)

        ratings = pd.read_excel(labelspath)
        filenames = ratings.groupby('Filename').size().index.tolist()
        self.labels = []
        for filename in filenames:
            rating = round(ratings[ratings['Filename'] == filename][
                           'Rating'].mean(), 2)
            self.labels.append({'Filename': filename, 'Rating': rating})
        self.labels = pd.DataFrame(self.labels)

    def __getitem__(self, index):

        img_path = self.imagespath[index % len(self.imagespath)]
        img = np.array(Image.open(img_path)) / 255.
        input_img = resize(img, (*self.img_shape, 3), mode='reflect')
        # array: (channel,height,width) -> tensor
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = torch.from_numpy(input_img).float()
        filename = img_path.split('/')[-1]
        label = self.labels.loc[self.labels[
            'Filename'] == filename, 'Rating'].values

        return img_path, input_img, label

    def __len__(self):

        return len(self.imagespath)


# TODO: train and test the model(resnet18)
