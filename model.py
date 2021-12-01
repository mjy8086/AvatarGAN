import os
import torch
from torch import nn
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
0. Backbone Encoder

마지막에 nn.AdaptiveAvgPool2d((1,1)) 한 다음에 -> 그래서 총 (channel, 1, 1)로 남기는듯
nn.Linear(차원, 2048)했어
"""


print('a')

print('a')

class CNNencoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.model(x)
        return out





# 11/29 SFCN과 비슷한 새로운 backbone
class Backbone_encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.conv1_1 = CNNencoder(in_channels, 32)
        self.conv1_2 = CNNencoder(32, 32)
        self.conv2_1 = CNNencoder(32, 64)
        self.conv2_2 = CNNencoder(64, 64)
        self.conv3_1 = CNNencoder(64, 128)
        self.conv3_2 = CNNencoder(128, 128)
        self.conv4_1 = CNNencoder(128, 256)
        self.conv4_2 = CNNencoder(256, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 2048)

    def forward(self, x):
        c1 = self.conv1_1(x)
        # (B, 16, 218, 182)
        c1 = self.conv1_2(c1)
        # (B, 16, 218, 182)
        p1 = self.pooling(c1)
        # (B, 16, 109, 91)
        c2 = self.conv2_1(p1)
        # (B, 32, 109, 91)
        c2 = self.conv2_2(c2)
        # (B, 32, 109, 91)
        p2 = self.pooling(c2)
        # (B, 32, 54, 45)
        c3 = self.conv3_1(p2)
        # (B, 64, 54, 45)
        c3 = self.conv3_2(c3)
        # (B, 64, 54, 45)
        p3 = self.pooling(c3)
        # (B, 64, 27, 22)
        c4 = self.conv4_1(p3)
        # (B, 128, 27, 22)
        c4 = self.conv4_2(c4)
        # (B, 128, 27, 22)
        c5 = self.pooling(c4)
        # (B, 128, 13, 11)
        # avg = self.avgpool(c5)
        # # (B, 128, 1, 1)
        # avg = avg.reshape(avg.size(0), -1)
        # # (B, 128)
        # out = self.fc(avg)
        # # (B, 2048)
        return c5


"""
0-1. Additional part

this part is for dividing backbone encoder from avg and last linear
"""

class Addition(nn.Module):
    # Residual Factorization Module
    def __init__(self):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 2048)

    def forward(self, x):

        avg = self.avgpool(x)
        # (B, 128, 1, 1)
        avg = avg.reshape(avg.size(0), -1)
        # (B, 128)
        out = self.fc(avg)
        # (B, 2048)
        return out




"""
1. DAL part
"""


class RFM(nn.Module):
    # Residual Factorization Module
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(2048, 2048)
                                 , nn.ReLU(inplace=True)
                                 , nn.Linear(2048, 2048)
                                 , nn.ReLU(inplace=True))

    def forward(self, xs):
        return self.seq(xs)


class DAL_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.RFM = RFM()

        """
        Age classifier for Classification
        """
        # self.age_classifier = nn.Sequential(nn.Linear(2048, 2048),
        #                                     nn.ReLU(inplace=True),
        #                                     nn.Linear(2048, 2048),
        #                                     nn.ReLU(inplace=True),
        #                                     nn.Linear(2048, 33))

        """
        Age classifier for Regression
        """
        # self.age_classifier = nn.Sequential(nn.Linear(2048, 2048),
                                            # nn.ReLU(inplace=True),
                                            # nn.Linear(2048, 1024),
                                            # nn.ReLU(inplace=True),
                                            # nn.Dropout(),
                                            # nn.Linear(1024, 1))
        """
        Age classifier for SFCN
        """
        self.age_classifier = nn.Sequential(nn.Linear(2048, 2048),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(),
                                            nn.Linear(2048, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(),
                                            nn.Linear(1024, 33))

        """
        Age classificer for Hybrid
        """
        self.age_classification = nn.Sequential(nn.Linear(2048, 2048),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(),
                                            nn.Linear(2048, 1024),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(),
                                            nn.Linear(1024, 33))

        self.age_regression = nn.Sequential(nn.Linear(33, 33),
                                            nn.ReLU(),
                                            nn.Dropout(),
                                            nn.Linear(33, 1))

    def forward(self, x):
        Age_feature = self.RFM(x)
        ID_feature = (x - Age_feature)

        # for Classification
        # age_pred = self.age_classifier(Age_feature)

        # for Regression
        # age_pred = self.age_classifier(Age_feature)
        # age_pred = torch.squeeze(age_pred)

        # for SFCN
        age_pred = list()
        a_c = self.age_classifier(Age_feature)
        a_s = F.log_softmax(a_c, dim=1)
        age_pred.append(a_s)

        # for Hybrid
        # a_c = list()
        # age_class = self.age_classification(Age_feature)
        # age_soft = F.log_softmax(age_class, dim=1)
        # a_c.append(age_soft)
        #
        # age_regress = self.age_regression(age_class)
        # a_r = torch.squeeze(age_regress)

        return ID_feature, age_pred
        # for Hybrid
        # return ID_feature, a_c, a_r


"""
2. SSL part
"""


class SSL(nn.Module):
    def __init__(self, dim=2048, pred_dim=512):
        super(SSL, self).__init__()

        self.backbone = Backbone_encoder()

        self.addition = Addition()

        self.DAL_model = DAL_model()

        self.projection = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(pred_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True),
                                        # self.projection,
                                        # 아래 layer 추가함
                                        nn.Linear(pred_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False))
        # self.projection[6].bias.requires_grad = False

        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(pred_dim, dim))

    def forward(self, x1, x2):

        b1 = self.backbone(x1)
        b2 = self.backbone(x2)

        a1 = self.addition(b1)
        a2 = self.addition(b2)

        d1, a_c = self.DAL_model(a1)
        # Hybrid
        # d1, a_c, a_r = self.DAL_model(a1)

        z1 = self.projection(d1)
        z2 = self.projection(a2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach(), a_c



