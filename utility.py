import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import cv2
import albumentations as A
import os



# set image path
Cartoon_path = '/home/mjy/AvatarGAN/data/cartoonset10k/cartoon/'
CelebA_path = '/home/mjy/AvatarGAN/data/source_train/data/'
test_CelebA_path = '/home/mjy/AvatarGAN/data/source_test/data/'


def create_df(path):
    name = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name.append(filename.split('.')[0])

    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))


df_cartoon = create_df(Cartoon_path)
df_celeb = create_df(CelebA_path)
df_test_celeb = create_df(test_CelebA_path)

Cartoon = df_cartoon['id'].values
CelebA = df_celeb['id'].values
test_CelebA = df_test_celeb['id'].values


# Dataset class
class Cartoon_Dataset(Dataset):
    def __init__(self, img_path, X, resize, mean, std):
        self.img_path = img_path
        self.X = X
        self.resize = resize
        self.mean = mean
        self.std = std
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        aug = self.resize(image=img)
        img = Image.fromarray(aug['image'])

        # for Gaussian Norm
        # t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        t = T.Compose([T.ToTensor()])
        img = t(img)

        return img

class CelebA_Dataset(Dataset):
    def __init__(self, img_path, X, resize, mean, std):
        self.img_path = img_path
        self.X = X
        self.resize = resize
        self.mean = mean
        self.std = std
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        aug = self.resize(image=img)
        img = Image.fromarray(aug['image'])

        # for Gaussian Norm
        # t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        t = T.Compose([T.ToTensor()])
        img = t(img)

        return img

Resize = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_NEAREST)])

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

# datasets
Cartoon_data_input = Cartoon_Dataset(Cartoon_path, Cartoon, Resize, mean, std)
Cartoon_data_target = Cartoon_Dataset(Cartoon_path, Cartoon, Resize, mean, std)

CelebA_data_input = CelebA_Dataset(CelebA_path, CelebA, Resize, mean, std)
CelebA_data_target = CelebA_Dataset(CelebA_path, CelebA, Resize, mean, std)


test_CelebA = CelebA_Dataset(test_CelebA_path, test_CelebA, Resize, mean, std)

# dataloader
# batch_size = 32

# Cartoon_loader = DataLoader(Cartoon_data, batch_size=batch_size, shuffle=True, drop_last=True)
# CelebA_loader = DataLoader(CelebA_data, batch_size=batch_size, shuffle=True, drop_last=True)
