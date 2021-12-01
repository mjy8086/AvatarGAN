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

from tqdm import tqdm
from model import *
from utility import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.empty_cache()


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=300)
# parser.add_argument('--d_iter', type=int, default=1)
# parser.add_argument('--g_iter', type=int, default=2)

args = parser.parse_args()


discriminator = Discriminator().cuda()
generator = ViT_UNet().cuda()



def main():

    batch_size, epoch = args.batch_size, args.epoch

    Cartoon_input_loader = DataLoader(Cartoon_data_input, batch_size=batch_size, shuffle=True, drop_last=True)
    Cartoon_target_loader = DataLoader(Cartoon_data_target, batch_size=batch_size, shuffle=True, drop_last=True)

    CelebA_input_loader = DataLoader(CelebA_data_input, batch_size=batch_size, shuffle=True, drop_last=True)
    CelebA_target_loader = DataLoader(CelebA_data_target, batch_size=batch_size, shuffle=True, drop_last=True)

    ####### Loss functions
    adversarial_loss = torch.nn.MSELoss().cuda()
    # classificiation_loss = torch.nn.NLLLoss().cuda()
    # classificiation_loss = torch.nn.CrossEntropyLoss().cuda()
    identity_loss = torch.nn.L1Loss().cuda()
    # Age_Prediction_loss = nn.MSELoss().cuda()



    ####### Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))


    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor

    ####### Training
    discriminator.train()
    generator.train()



    # for epoch in tqdm(range(0, epoch), desc='Epoch'):
    for epoch in range(epoch):

        print('Epoch {}/{}'.format(epoch, args.epoch))

        Cartoon_D_loss = 0
        Cartoon_G_loss = 0
        CelebA_D_loss = 0
        CelebA_G_loss = 0

        for batch, (cartoon_input_data, cartoon_target_data, celeba_input_data, celeba_target_data) in tqdm(enumerate(zip(Cartoon_input_loader, Cartoon_target_loader, CelebA_input_loader, CelebA_target_loader)), total=len(Cartoon_input_loader)):

            cartoon_input = cartoon_input_data
            cartoon_target = cartoon_target_data

            celeba_input = celeba_input_data
            celeba_target = celeba_target_data

            cartoon_input = Variable(cartoon_input).cuda()
            cartoon_target = Variable(cartoon_target).cuda()

            celeba_input = Variable(celeba_input).cuda()
            celeba_target = Variable(celeba_target).cuda()



            """
            Cartoon dataset train
                        
            """



            """
            Train Discriminator
            """

            optimizer_D.zero_grad()

            # Real Loss
            pred_real = discriminator(cartoon_target)
            real = Variable(torch.ones(pred_real.size()))
            loss_D_real = adversarial_loss(pred_real.cuda(), real.cuda())

            loss_D_real.backward()
            optimizer_D.step()


            # Fake Loss
            fake_img = generator(cartoon_input)

            pred_fake = discriminator(fake_img.detach())
            fake = Variable(torch.zeros(pred_fake.size()))
            loss_D_fake = adversarial_loss(pred_fake.cuda(), fake.cuda())

            loss_D_fake.backward()
            optimizer_D.step()


            """
            Train Generator
            """

            optimizer_G.zero_grad()

            fake_img = generator(cartoon_input)

            pred_real2 = discriminator(fake_img)
            loss_G_real = adversarial_loss(pred_real2, real.cuda())

            ## Identity Loss
            loss_G_identity = identity_loss(cartoon_input, fake_img)

            # Total Loss
            loss_G = loss_G_real + 0.001 * loss_G_identity

            loss_G.backward()
            optimizer_G.step()


            Cartoon_D_loss += loss_D_fake.item()
            Cartoon_G_loss += loss_G_real.item()




            """
            CelebA dataset train

            """

            """
            Train Discriminator
            """

            optimizer_D.zero_grad()

            # Real Loss
            pred_real = discriminator(celeba_target)
            real = Variable(torch.ones(pred_real.size()))
            loss_D_real = adversarial_loss(pred_real.cuda(), real.cuda())

            loss_D_real.backward()
            optimizer_D.step()

            # Fake Loss
            fake_img = generator(celeba_input)

            pred_fake = discriminator(fake_img.detach())
            fake = Variable(torch.zeros(pred_fake.size()))
            loss_D_fake = adversarial_loss(pred_fake.cuda(), fake.cuda())

            loss_D_fake.backward()
            optimizer_D.step()

            """
            Train Generator
            """

            optimizer_G.zero_grad()

            fake_img = generator(celeba_input)

            pred_real2 = discriminator(fake_img)
            loss_G_real = adversarial_loss(pred_real2, real.cuda())

            ## Identity Loss
            loss_G_identity = identity_loss(celeba_input, fake_img)

            # Total Loss
            loss_G = loss_G_real + 0.001 * loss_G_identity

            loss_G.backward()
            optimizer_G.step()

            CelebA_D_loss += loss_D_fake.item()
            CelebA_G_loss += loss_G_real.item()




        print('Cartoon_D_loss: {:3f} Cartoon_G_loss: {:3f}'.format(Cartoon_D_loss / len(Cartoon_input_loader),
                                                                   Cartoon_G_loss / len(Cartoon_input_loader)))
        print('CelebA_D_loss: {:3f} CelebA_G_loss: {:3f}'.format(CelebA_D_loss / len(Cartoon_input_loader),
                                                                 CelebA_G_loss / len(Cartoon_input_loader)))

        # if epoch % 5 == 0:
        #
        #     createFolder('/DataCommon2/mjy/IBSSL/output_image/epoch_%d' % epoch)
        #
        #     # torch.save(generator.state_dict(), '/DataCommon2/mjy/IBSSL/GAN_save/G_lr_%d.pth' % epoch)
        #
        #     generator.train()




if __name__== '__main__':
    main()