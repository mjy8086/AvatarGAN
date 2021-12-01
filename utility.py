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

# the number of classes of this datase is 23
n_classes = 23


def create_df(path):
    name = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name.append(filename.split('.')[0])

    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))


df_cartoon = create_df(Cartoon_path)
df_celeb = create_df(CelebA_path)

Cartoon = df_cartoon['id'].values
CelebA = df_cartoon['id'].values


# split data
# X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)
# X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

Cartoon_img = Image.open(Cartoon_path + df_cartoon['id'][100] + '.png')
CelebA_img = Image.open(CelebA_path + df_celeb['id'][100] + '.jpg')


# Dataset class
class Cartoon_Dataset(Dataset):
    def __init__(self, img_path, X, resize):
        self.img_path = img_path
        self.X = X
        self.resize = resize
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        aug = self.resize(image=img)
        img = Image.fromarray(aug['image'])

        img = Image.fromarray(img)
        return img

class CelebA_Dataset(Dataset):
    def __init__(self, img_path, X, resize):
        self.img_path = img_path
        self.X = X
        self.resize = resize
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        aug = self.resize(image=img)
        img = Image.fromarray(aug['image'])

        img = Image.fromarray(img)
        return img

Resize = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_NEAREST)])

# datasets
Cartoon_data = Cartoon_Dataset(Cartoon_path, Cartoon, Resize)
CelebA_data = CelebA_Dataset(CelebA_path, CelebA)

# dataloader
batch_size = 32

Cartoon_loader = DataLoader(Cartoon_data, batch_size=batch_size, shuffle=True, drop_last=True)
CelebA_loader = DataLoader(CelebA_data, batch_size=batch_size, shuffle=True, drop_last=True)
