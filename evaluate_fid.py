import os
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm, Flatten, Conv2d
import torchvision
from torchvision import transforms, models
import math
import copy
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from itertools import cycle
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm
from pytorch_fid.inception import InceptionV3


from tqdm import tqdm
from model import *
from utility import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()

model = InceptionV3().cuda()


CelebA_Encoder_path = "/home/mjy/AvatarGAN/generated_img/epoch_40/CelebA_Encoder.pth"
Bottleneck_path = "/home/mjy/AvatarGAN/generated_img/epoch_40/Bottleneck.pth"
Cartoon_Decoder_path = "/home/mjy/AvatarGAN/generated_img/epoch_40/Cartoon_Decoder.pth"



CelebA_Encoder = CelebA_Encoder().cuda()
CelebA_Encoder.load_state_dict(torch.load(CelebA_Encoder_path))
CelebA_Encoder.cuda()

Bottleneck = Bottleneck().cuda()
Bottleneck.load_state_dict(torch.load(Bottleneck_path))
Bottleneck.cuda()

Cartoon_Decoder = Cartoon_Decoder().cuda()
Cartoon_Decoder.load_state_dict(torch.load(Cartoon_Decoder_path))
Cartoon_Decoder.cuda()


for param in CelebA_Encoder.parameters():
    param.requires_grad_(False)
for param in Bottleneck.parameters():
    param.requires_grad_(False)
for param in Cartoon_Decoder.parameters():
    param.requires_grad_(False)




def main():
    batch_size = args.batch_size

    test_CelebA_loader = DataLoader(test_CelebA, batch_size=batch_size, shuffle=False, drop_last=True)

    CelebA_Encoder.eval()
    Bottleneck.eval()
    Cartoon_Decoder.eval()


    FID_score = 0


    with torch.no_grad():

        for batch, test_data in tqdm(enumerate(test_CelebA_loader), total=len(test_CelebA_loader)):
            test_img = test_data

            test_img = Variable(test_img).cuda()


            e_out, c5, c4, c3, c2, c1 = CelebA_Encoder(test_img)
            b_out = Bottleneck(e_out)
            fake_img = Cartoon_Decoder(b_out, c5, c4, c3, c2, c1)

            fid = calculate_fretchet(test_img, fake_img.cuda(), model, device='cuda')

            FID_score += fid.item()

    print('FID_score: {:3f}'.format(FID_score / len(test_CelebA_loader)))




if __name__ == '__main__':
    main()