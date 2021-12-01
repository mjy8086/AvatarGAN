import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from model import *
from utility import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

    Cartoon_loader = DataLoader(Cartoon_data, batch_size=batch_size, shuffle=True, drop_last=True)
    CelebA_loader = DataLoader(CelebA_data, batch_size=batch_size, shuffle=True, drop_last=True)

    ####### Loss functions
    adversarial_loss = torch.nn.MSELoss().cuda()
    # classificiation_loss = torch.nn.NLLLoss().cuda()
    # classificiation_loss = torch.nn.CrossEntropyLoss().cuda()
    # identity_loss = torch.nn.L1Loss().cuda()
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

        D_loss = 0
        G_loss = 0
        A_loss = 0
        Accuracy = 0
        MAE = 0

        for batch, (input_data, target_data) in tqdm(enumerate(zip(Cartoon_loader, CelebA_loader)), total=len(Cartoon_loader)):

            input_img, input_age = input_data
            real_img, target_age = target_data

            real_img = Variable(real_img).cuda()
            # target_age = torch.from_numpy(target_age)
            # target_age = Variable(target_age).cuda()
            target_age = Variable(target_age).long()
            target_age = target_age.cuda()
            input_img = Variable(input_img).cuda()
            input_age = Variable(input_age).long()
            input_age = input_age.cuda()
            # input_age_pred = input_age_pred.cuda()
            # target_age_pred = target_age_pred.cuda()


            """
            Train Discriminator
            """
            # for d_iter in range(args.d_iter):

            optimizer_D.zero_grad()

            # Real Loss
            pred_real = discriminator(real_img)
            real = Variable(torch.ones(pred_real.size()))
            loss_D_real = adversarial_loss(pred_real.cuda(), real.cuda())
            # loss_D_real_class = classificiation_loss(pred_age.cuda(), target_age)
            # loss_D_real = loss_D_real_real

            loss_D_real.backward()
            optimizer_D.step()

            # accuracy = compute_acc(pred_age, target_age)

            # Fake Loss
            target_age2 = torch.randint(low=0, high=32, size=(args.batch_size,))
            target_age2 = target_age2.cuda()

            input_age2 = input_age - 48

            diff_age = target_age2 - input_age2 + 32 # +6이었어 원래

            diff_age.cuda()

            fake_MRI, _ = generator(input_img, diff_age)

            pred_fake = discriminator(fake_MRI.detach())
            fake = Variable(torch.zeros(pred_fake.size()))
            loss_D_fake = adversarial_loss(pred_fake.cuda(), fake.cuda())
            # loss_D_fake_class = classificiation_loss(pred_age2.cuda(), target_age2)
            # loss_D_fake = loss_D_fake_fake

            loss_D_fake.backward()
            optimizer_D.step()


            """
            Train Generator
            """
            # for g_iter in range(args.g_iter):

            optimizer_G.zero_grad()

            fake_MRI, fake_pred_age = generator(input_img, diff_age)

            pred_real2 = discriminator(fake_MRI)
            loss_G_real = adversarial_loss(pred_real2, real.cuda())
            # loss_G_class = classificiation_loss(pred_age3, target_age2)

            ## Identity Loss
            # loss_G_identity = identity_loss(input_img, fake_MRI)

            ## Age Prediction Loss
            real_pred_age = input_age2 + diff_age - 32 + 48 # diff_age는 이미 +32한거니 -32해야 정상 나이
            # fake_pred_age = SFCN(fake_MRI)
            loss_G_prediction = Age_Prediction_loss(fake_pred_age.float(), real_pred_age.float())

            mae = torch.abs(fake_pred_age - real_pred_age).sum() / batch_size

            # Total Loss
            # loss_G = 75*loss_G_real + 30*loss_G_prediction
            loss_G = loss_G_real + 10 * loss_G_prediction

            loss_G.backward()
            optimizer_G.step()


            D_loss += loss_D_fake.item()
            G_loss += loss_G_real.item()
            A_loss += loss_G_prediction.item()
            # Accuracy += accuracy
            MAE += mae.item()

            # print(
            #       "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Accuracy: %d]"
            #       % (epoch, args.epoch, batch, len(Input_dataloader), loss_D_fake.item(), loss_G.item(), accuracy)
            #       )






        print('D_loss: {:3f} G_loss: {:3f} A_loss: {:3f} Gen_MAE: {:.3f}'.format(D_loss / len(Input_dataloader),
                                                                                        G_loss / len(Input_dataloader),
                                                                                        A_loss / len(Input_dataloader),
                                                                                        MAE / len(Input_dataloader)))

        if epoch % 5 == 0:

            createFolder('/DataCommon2/mjy/IBSSL/output_image/epoch_%d' % epoch)

            # torch.save(generator.state_dict(), '/DataCommon2/mjy/IBSSL/GAN_save/G_lr_%d.pth' % epoch)

            generator.train()




if __name__== '__main__':
    main()