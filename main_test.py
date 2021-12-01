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
from model_test import *
from utility import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=300)
# parser.add_argument('--d_iter', type=int, default=1)
# parser.add_argument('--g_iter', type=int, default=2)

args = parser.parse_args()

Cartoon_Encoder = Cartoon_Encoder().cuda()
CelebA_Encoder = CelebA_Encoder().cuda()

Bottleneck = Bottleneck().cuda()

Cartoon_Decoder = Cartoon_Decoder().cuda()
CelebA_Decoder = CelebA_Decoder().cuda()

Cartoon_Discriminator = Cartoon_Discriminator().cuda()
CelebA_Discriminator = CelebA_Discriminator().cuda()


"""
Making directory
"""

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)



def main():
    batch_size, epoch = args.batch_size, args.epoch

    Cartoon_input_loader = DataLoader(Cartoon_data_input, batch_size=batch_size, shuffle=True, drop_last=True)
    Cartoon_target_loader = DataLoader(Cartoon_data_target, batch_size=batch_size, shuffle=True, drop_last=True)

    CelebA_input_loader = DataLoader(CelebA_data_input, batch_size=batch_size, shuffle=True, drop_last=True)
    CelebA_target_loader = DataLoader(CelebA_data_target, batch_size=batch_size, shuffle=True, drop_last=True)

    test_CelebA_loader = DataLoader(test_CelebA, batch_size=batch_size, shuffle=True, drop_last=True)

    ####### Loss functions
    adversarial_loss = torch.nn.MSELoss().cuda()
    # classificiation_loss = torch.nn.NLLLoss().cuda()
    # classificiation_loss = torch.nn.CrossEntropyLoss().cuda()
    identity_loss = torch.nn.L1Loss().cuda()
    # Age_Prediction_loss = nn.MSELoss().cuda()

    ####### Optimizers
    optimizer_cartoon_G = torch.optim.Adam(list(Cartoon_Encoder.parameters()) + list(Bottleneck.parameters()) + list(Cartoon_Decoder.parameters()), lr=0.001, betas=(0.5, 0.999))
    optimizer_cartoon_D = torch.optim.Adam(Cartoon_Discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

    optimizer_celeba_G = torch.optim.Adam(list(CelebA_Encoder.parameters()) + list(Bottleneck.parameters()) + list(CelebA_Decoder.parameters()), lr=0.001, betas=(0.5, 0.999))
    optimizer_celeba_D = torch.optim.Adam(CelebA_Discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))


    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor

    ####### Training
    Cartoon_Encoder.train()
    CelebA_Encoder.train()
    Bottleneck.train()
    Cartoon_Decoder.train()
    CelebA_Decoder.train()
    Cartoon_Discriminator.train()
    CelebA_Discriminator.train()


    for epoch in range(epoch):

        print('Epoch {}/{}'.format(epoch, args.epoch))

        Cartoon_D_loss = 0
        Cartoon_G_loss = 0
        CelebA_D_loss = 0
        CelebA_G_loss = 0

        for batch, (cartoon_input_data, cartoon_target_data, celeba_input_data, celeba_target_data) in tqdm(
                enumerate(zip(Cartoon_input_loader, Cartoon_target_loader, CelebA_input_loader, CelebA_target_loader)),
                total=len(Cartoon_input_loader)):
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

            optimizer_cartoon_D.zero_grad()

            # Real Loss
            pred_real = Cartoon_Discriminator(cartoon_target)
            real = Variable(torch.ones(pred_real.size()))
            loss_D_real = adversarial_loss(pred_real.cuda(), real.cuda())

            loss_D_real.backward()
            optimizer_cartoon_D.step()

            # Fake Loss
            e_out, c5, c4, c3, c2, c1 = Cartoon_Encoder(cartoon_input)
            b_out = Bottleneck(e_out)
            fake_img = Cartoon_Decoder(b_out, c5, c4, c3, c2, c1)


            pred_fake = Cartoon_Discriminator(fake_img.detach())
            fake = Variable(torch.zeros(pred_fake.size()))
            loss_D_fake = adversarial_loss(pred_fake.cuda(), fake.cuda())

            loss_D_fake.backward()
            optimizer_cartoon_D.step()

            """
            Train Generator
            """

            optimizer_cartoon_G.zero_grad()

            e_out, c5, c4, c3, c2, c1 = Cartoon_Encoder(cartoon_input)
            b_out = Bottleneck(e_out)
            fake_img = Cartoon_Decoder(b_out, c5, c4, c3, c2, c1)

            pred_real2 = Cartoon_Discriminator(fake_img)
            loss_G_real = adversarial_loss(pred_real2, real.cuda())

            ## Identity Loss
            loss_G_identity = identity_loss(cartoon_input, fake_img)

            # Total Loss
            loss_G = loss_G_real + 0.001 * loss_G_identity

            loss_G.backward()
            optimizer_cartoon_G.step()

            Cartoon_D_loss += loss_D_fake.item()
            Cartoon_G_loss += loss_G_real.item()

            """
            CelebA dataset train

            """

            """
            Train Discriminator
            """

            optimizer_celeba_D.zero_grad()

            # Real Loss
            pred_real = CelebA_Discriminator(celeba_target)
            real = Variable(torch.ones(pred_real.size()))
            loss_D_real = adversarial_loss(pred_real.cuda(), real.cuda())

            loss_D_real.backward()
            optimizer_celeba_D.step()

            # Fake Loss
            e_out, c5, c4, c3, c2, c1 = CelebA_Encoder(celeba_input)
            b_out = Bottleneck(e_out)
            fake_img = CelebA_Decoder(b_out, c5, c4, c3, c2, c1)

            pred_fake = CelebA_Discriminator(fake_img.detach())
            fake = Variable(torch.zeros(pred_fake.size()))
            loss_D_fake = adversarial_loss(pred_fake.cuda(), fake.cuda())

            loss_D_fake.backward()
            optimizer_celeba_D.step()

            """
            Train Generator
            """

            optimizer_celeba_G.zero_grad()

            e_out, c5, c4, c3, c2, c1 = CelebA_Encoder(celeba_input)
            b_out = Bottleneck(e_out)
            fake_img = CelebA_Decoder(b_out, c5, c4, c3, c2, c1)

            pred_real2 = CelebA_Discriminator(fake_img)
            loss_G_real = adversarial_loss(pred_real2, real.cuda())

            ## Identity Loss
            loss_G_identity = identity_loss(celeba_input, fake_img)

            # Total Loss
            loss_G = loss_G_real + 0.001 * loss_G_identity

            loss_G.backward()
            optimizer_celeba_G.step()

            CelebA_D_loss += loss_D_fake.item()
            CelebA_G_loss += loss_G_real.item()

        print('Cartoon_D_loss: {:3f} Cartoon_G_loss: {:3f}'.format(Cartoon_D_loss / len(Cartoon_input_loader),
                                                                   Cartoon_G_loss / len(Cartoon_input_loader)))
        print('CelebA_D_loss: {:3f} CelebA_G_loss: {:3f}'.format(CelebA_D_loss / len(Cartoon_input_loader),
                                                                 CelebA_G_loss / len(Cartoon_input_loader)))


        # if epoch % 1 == 0:
        Cartoon_Encoder.eval()
        CelebA_Encoder.eval()
        Bottleneck.eval()
        Cartoon_Decoder.eval()
        CelebA_Decoder.eval()
        Cartoon_Discriminator.eval()
        CelebA_Discriminator.eval()

        # create a folder for generated images
        createFolder('/home/mjy/AvatarGAN/generated_img/epoch_%d' % epoch)

        with torch.no_grad():

            for batch, test_data in tqdm(enumerate(test_CelebA_loader), total=len(test_CelebA_loader)):
                test_img = test_data
                test_img = Variable(test_img).cuda()

                # generate syntesized image
                out_cartoon_encoder = Cartoon_Encoder(test_img)
                out_bottleneck = Bottleneck(out_cartoon_encoder)
                generated_image = CelebA_Decoder(out_bottleneck)

                generated_image = generated_image.permute(0, 2, 3, 1)

                output = generated_image.detach()
                output = output.cpu()
                output = np.squeeze(output)
                output = output.numpy()

                output0 = output[0]
                output1 = output[1]
                output2 = output[2]
                output3 = output[3]
                output4 = output[4]
                output5 = output[5]
                output6 = output[6]
                output7 = output[7]

                plt.axis('off'), plt.xticks([]), plt.yticks([])
                plt.tight_layout()
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

                plt.imshow(output0)
                plt.savefig('/home/mjy/AvatarGAN/generated_img/epoch_%d/output0.png' % epoch, bbox_inches='tight',
                            pad_inches=0)
                plt.imshow(output1)
                plt.savefig('/home/mjy/AvatarGAN/generated_img/epoch_%d/output1.png' % epoch, bbox_inches='tight',
                            pad_inches=0)
                plt.imshow(output2)
                plt.savefig('/home/mjy/AvatarGAN/generated_img/epoch_%d/output2.png' % epoch, bbox_inches='tight',
                            pad_inches=0)
                plt.imshow(output3)
                plt.savefig('/home/mjy/AvatarGAN/generated_img/epoch_%d/output3.png' % epoch, bbox_inches='tight',
                            pad_inches=0)
                plt.imshow(output4)
                plt.savefig('/home/mjy/AvatarGAN/generated_img/epoch_%d/output4.png' % epoch, bbox_inches='tight',
                            pad_inches=0)
                plt.imshow(output5)
                plt.savefig('/home/mjy/AvatarGAN/generated_img/epoch_%d/output5.png' % epoch, bbox_inches='tight',
                            pad_inches=0)
                plt.imshow(output6)
                plt.savefig('/home/mjy/AvatarGAN/generated_img/epoch_%d/output6.png' % epoch, bbox_inches='tight',
                            pad_inches=0)
                plt.imshow(output7)
                plt.savefig('/home/mjy/AvatarGAN/generated_img/epoch_%d/output7.png' % epoch, bbox_inches='tight',
                            pad_inches=0)

        Cartoon_Encoder.train()
        CelebA_Encoder.train()
        Bottleneck.train()
        Cartoon_Decoder.train()
        CelebA_Decoder.train()
        Cartoon_Discriminator.train()
        CelebA_Discriminator.train()

        if epoch % 1 == 0:

            torch.save(Cartoon_Encoder.state_dict(), '/home/mjy/AvatarGAN/generated_img/epoch_%d/Cartoon_Encoder_%d.pth' % epoch)
            torch.save(CelebA_Encoder.state_dict(), '/home/mjy/AvatarGAN/generated_img/epoch_%d/CelebA_Encoder_%d.pth' % epoch)
            torch.save(Bottleneck.state_dict(), '/home/mjy/AvatarGAN/generated_img/epoch_%d/Bottleneck_%d.pth' % epoch)
            torch.save(Cartoon_Decoder.state_dict(), '/home/mjy/AvatarGAN/generated_img/epoch_%d/Cartoon_Decoder_%d.pth' % epoch)
            torch.save(CelebA_Decoder.state_dict(), '/home/mjy/AvatarGAN/generated_img/epoch_%d/CelebA_Decoder_%d.pth' % epoch)
            torch.save(Cartoon_Discriminator.state_dict(), '/home/mjy/AvatarGAN/generated_img/epoch_%d/Cartoon_Discriminator_%d.pth' % epoch)
            torch.save(CelebA_Discriminator.state_dict(), '/home/mjy/AvatarGAN/generated_img/epoch_%d/CelebA_Discriminator_%d.pth' % epoch)


if __name__ == '__main__':
    main()