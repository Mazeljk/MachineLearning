import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from skimage.transform import resize
import torchvision
import torch.nn as nn
import time


class Config:

    imagespath = './SCUT-FBP5500_v2/Images/'
    labelspath = './SCUT-FBP5500_v2/All_Ratings.xlsx'
    img_shape = (224, 224)
    is_shuffle = True
    modeldir = './model'
    use_cuda = False
    gpus = ''  # the ids of gpu
    ngpus = 0     # the number of used
    batch_size = 64
    workers = 4
    num_epochs = 50
    save_interval = 5
    error_torlerance = 0.5
    logfile = 'train.log'


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


class DeepNN():

    def __init__(self, config):

        self.Config = config

    def train(self):

        if not os.path.exists(self.Config.modeldir):
            os.mkdir(self.Config.modeldir)
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        use_cuda = torch.cuda.is_available() and self.Config.use_cuda
        if use_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.Config.gpus
            if self.Config.ngpus > 1:
                model = nn.DataParallel(model).cuda()
            else:
                model = model.cuda()
        model.train()
        dataloader = torch.utils.data.DataLoader(Images(self.Config.imagespath,
                                                        self.Config.labelspath,
                                                        self.Config.img_shape,
                                                        self.Config.is_shuffle,
                                                        'train'),
                                                 batch_size=self.Config.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.Config.workers)
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()))
        criterion = nn.MSELoss()
        for epoch in range(1, self.Config.num_epochs + 1):
            self.Logging('[INFO]: epoch now is %d...' %
                         epoch, self.Config.logfile)
            for batch_i, (_, imgs, targets) in enumerate(dataloader):
                imgs = imgs.type(FloatTensor)
                targets = targets.type(FloatTensor)
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, targets)
                if self.Config.ngpus > 1:
                    loss = loss.sum()
                self.Logging('[INFO]: batch%d of epoch%d, loss is %.2f...' %
                             (batch_i, epoch, loss.item()))
                loss.backward()
                optimizer.step()
            if (epoch % self.Config.save_interval == 0) and (epoch > 0):
                pklpath = os.path.join(
                    self.Config.modeldir, 'epoch_%s.pkl' % str(epoch))
                if self.Config.ngpus > 1:
                    cur_model = model.module
                else:
                    cur_model = model
                torch.save(cur_model.state_dict(), pklpath)
                acc = self.test(model)
                self.Logging('[INFO]: Accuracy of epoch %d is %.2f...' %
                             (epoch, acc), self.Config.logfile)

    def test(self, model):

        model.eval()
        dataloader = torch.utils.data.DataLoader(Images(self.Config.imagespath,
                                                        self.Config.labelspath,
                                                        self.Config.img_shape,
                                                        self.Config.is_shuffle,
                                                        'test'),
                                                 batch_size=self.Config.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.Config.workers)
        use_cuda = torch.cuda.is_available() and self.Config.use_cuda
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        n_correct = 0
        n_total = 0
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            imgs = imgs.type(FloatTensor)
            targets = targets.type(FloatTensor)
            preds = model(imgs)
            n_correct += (abs(targets - preds) <
                          self.Config.error_torlerance).sum().item()
            n_total += imgs.size(0)
        acc = n_correct / n_total
        model.train()
        return acc

    def Logging(self, message, savefile=None):

        content = '%s %s' % (time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime()), message)
        if savefile:
            f = open(savefile, 'a')
            f.write(content + '\n')
            f.close()
        print(content)


if __name__ == '__main__':
    config = Config()
    DNN = DeepNN(config)
    DNN.train()
