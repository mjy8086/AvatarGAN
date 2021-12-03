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
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3


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




"""
FID score
"""


def calculate_activation_statistics(images, model, device=None, batch_size=128, dims=2048):
    model.eval()
    act = np.empty((len(images), dims))

    if device is None:
        batch = images
    else:
        batch = images.to(device)
    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma




def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)





def calculate_fretchet(images_real, images_fake, model, device):
    mu_1, std_1 = calculate_activation_statistics(images_real, model, device=device)
    mu_2, std_2 = calculate_activation_statistics(images_fake, model, device=device)

    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value