import torch
from torch import nn
import torch.nn.functional
from torch.nn import Dropout, Softmax, Linear, LayerNorm, Conv2d
import math
import copy
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


"""
ViT_UNet is consists of 3 parts
1) CNN Encoder
2) bottle neck Vision Trnasformer(ViT)
3) CNN Decoder
"""



# CNN Encoder

# BatchNorm is not used because it's not good for ViT
# GroupNorm and LayerNorm is used instead

# Conv with GroupNorm
class CNNencoder_gn(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class CNNencoder_gn_drop(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        out = self.model(x)
        return out

# Conv with LayerNorm
class CNNencoder_ln(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(3, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out


# CNN Concat with GroupNorm
class Concat_gn(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, skip):

        x = torch.cat((x, skip), 1)
        out = self.model(x)
        return out

# CNN concat with LayerNorm
class Concat_ln(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(3, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, skip):

        x = torch.cat((x, skip), 1)
        out = self.model(x)
        return out



# Making ViT

# Patch Embedding

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size):
        super(Embeddings, self).__init__()
        down_factor = 4
        # input image가 얼마나 많이 pooling을 거치냐가 down_factor
        # Maxpool2d가 4번 있으니 down_factor = 4
        patch_size = (2, 2)
        # patch_size는 2로 설정
        n_patches = int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]))
        # n_pathces = (512/2**4//8) * (768/2**4//8) = 4
        self.patch_embeddings = Conv2d(in_channels=256,
                                       # 우선 in channels는 128로 설정하자
                                       out_channels=768,
                                       # out_channels = hidden size D = 768
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))

        self.dropout = Dropout(0.1)

    def forward(self, x):
        # input = (B, 256, 48, 32)
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        # (B, 768, 24, 16)
        x = x.flatten(2)
        # (B, 768, 384)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # (B, 384, 768)
        position_embeddings = self.position_embeddings
        # position_embeddings = (B, 384, 768)
        embeddings = x + position_embeddings
        # (B, 384, 768)
        embeddings = self.dropout(embeddings)
        return embeddings


# Multi-head self attention (MSA) - layer norm not included

class MSA(nn.Module):
    def __init__(self):
        super(MSA, self).__init__()
        self.num_attention_heads = 12
        # Number of head = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        # Attention Head size = Hidden size(D)(768) / Number of head(12) = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # All Head size = (12 * 64) = 768 = Hidden size
        self.query = Linear(768, self.all_head_size)
        self.key = Linear(768, self.all_head_size)
        self.value = Linear(768, self.all_head_size)
        self.out = Linear(768, 768)
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        x = x.view([x.size()[0], -1, self.num_attention_heads, self.attention_head_size])
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


# MLP - layer norm not included

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = Linear(768, 3072)
        self.fc2 = Linear(3072, 768)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# Block - incorporating MSA, MLP, Layer Norm

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = LayerNorm(768, eps=1e-6)
        self.ffn_norm = LayerNorm(768, eps=1e-6)
        self.ffn = MLP()
        self.attn = MSA()

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


#  ViTencoder - ViT Encoder with Blocks

class ViTencoder(nn.Module):
    def __init__(self):
        super(ViTencoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(768, eps=1e-6)
        for _ in range(12):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):


        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded


#  ViT 마지막에 나온 latent를 CNNdecoder에 넣기 위해 변환시키기위한 Conv

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_groupnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_groupnorm),
        )
        relu = nn.LeakyReLU(inplace=True)

        gn = nn.GroupNorm(16, out_channels, eps=1e-6)

        super(Conv2dReLU, self).__init__(conv, gn, relu)


#  ViT

class ViT(nn.Module):
    def __init__(self, img_size):
        super(ViT, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        self.encoder = ViTencoder()
        self.img_size = img_size
        self.patch_size = (2, 2)
        self.down_factor = 4
        self.conv_more = Conv2dReLU(768, 256, kernel_size=3, padding=1, use_groupnorm=True)

    def forward(self, x):
        # (B, 256, 16, 16)
        x = self.embeddings(x)
        # (B, 64, 768)
        x = self.encoder(x)  # (B, n_patch, hidden)
        # (B, 64, 768)
        B, n_patch, hidden = x.size()
        # B=B, n_patch=64, hidden=768
        h, w = (self.img_size[0] // 2**self.down_factor // self.patch_size[0]), (self.img_size[1] // 2**self.down_factor // self.patch_size[0])
        # h=8, w=8
        x = x.permute(0, 2, 1)
        # (B, 768, 64)
        x = x.contiguous().view(B, hidden, h, w)
        # (B, 768, 8, 8)
        x = self.conv_more(x)
        # (B, 256, 8, 8)
        return x





"""
Encoder
"""
class Cartoon_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.conv1_1 = CNNencoder_gn(3, 16)
        self.conv1_2 = CNNencoder_gn(16, 16)
        self.conv2_1 = CNNencoder_gn(16, 32)
        self.conv2_2 = CNNencoder_gn(32, 32)
        self.conv3_1 = CNNencoder_gn(32, 64)
        self.conv3_2 = CNNencoder_gn(64, 64)
        self.conv4_1 = CNNencoder_gn(64, 128)
        self.conv4_2 = CNNencoder_gn(128, 128)
        self.conv5_1 = CNNencoder_gn(128, 256)
        self.conv5_2 = CNNencoder_gn(256, 256)
    def forward(self, x):
        c1 = self.conv1_1(x)
        c1 = self.conv1_2(c1)
        p1 = self.pooling(c1)
        c2 = self.conv2_1(p1)
        c2 = self.conv2_2(c2)
        p2 = self.pooling(c2)
        c3 = self.conv3_1(p2)
        c3 = self.conv3_2(c3)
        p3 = self.pooling(c3)
        c4 = self.conv4_1(p3)
        c4 = self.conv4_2(c4)
        p4 = self.pooling(c4)
        c5 = self.conv5_1(p4)
        out = self.conv5_2(c5)
        return out


class CelebA_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.conv1_1 = CNNencoder_gn(3, 16)
        self.conv1_2 = CNNencoder_gn(16, 16)
        self.conv2_1 = CNNencoder_gn(16, 32)
        self.conv2_2 = CNNencoder_gn(32, 32)
        self.conv3_1 = CNNencoder_gn(32, 64)
        self.conv3_2 = CNNencoder_gn(64, 64)
        self.conv4_1 = CNNencoder_gn(64, 128)
        self.conv4_2 = CNNencoder_gn(128, 128)
        self.conv5_1 = CNNencoder_gn(128, 256)
        self.conv5_2 = CNNencoder_gn(256, 256)
    def forward(self, x):
        c1 = self.conv1_1(x)
        c1 = self.conv1_2(c1)
        p1 = self.pooling(c1)
        c2 = self.conv2_1(p1)
        c2 = self.conv2_2(c2)
        p2 = self.pooling(c2)
        c3 = self.conv3_1(p2)
        c3 = self.conv3_2(c3)
        p3 = self.pooling(c3)
        c4 = self.conv4_1(p3)
        c4 = self.conv4_2(c4)
        p4 = self.pooling(c4)
        c5 = self.conv5_1(p4)
        out = self.conv5_2(c5)
        return out





"""
Bottleneck
"""

class Bottleneck(nn.Module):
    def __init__(self, img_size=(256, 256)):
        super().__init__()
        self.vit = ViT(img_size)

    def forward(self, x):
        out = self.vit(x)
        return out


"""
Decoder
"""

class Cartoon_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.concat1 = CNNencoder_gn(256, 128)
        self.convup1 = CNNencoder_gn(128, 128)
        self.concat2 = CNNencoder_gn(128, 64)
        self.convup2 = CNNencoder_gn(64, 64)
        self.concat3 = CNNencoder_gn(64, 32)
        self.convup3 = CNNencoder_gn(32, 32)
        self.concat4 = CNNencoder_gn(32, 16)
        self.convup4 = CNNencoder_gn(16, 16)
        self.concat5 = CNNencoder_ln(16, 3)
        self.convup5 = CNNencoder_ln(3, 3)
    def forward(self, x):
        v1 = self.upsample(x)
        u1 = self.concat1(v1)
        u1 = self.convup1(u1)
        u1 = self.upsample(u1)
        u2 = self.concat2(u1)
        u2 = self.convup2(u2)
        u2 = self.upsample(u2)
        u3 = self.concat3(u2)
        u3 = self.convup3(u3)
        u3 = self.upsample(u3)
        u4 = self.concat4(u3)
        u4 = self.convup4(u4)
        u4 = self.upsample(u4)
        u5 = self.concat5(u4)
        out = self.convup5(u5)
        return out


class CelebA_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.concat1 = CNNencoder_gn(256, 128)
        self.convup1 = CNNencoder_gn(128, 128)
        self.concat2 = CNNencoder_gn(128, 64)
        self.convup2 = CNNencoder_gn(64, 64)
        self.concat3 = CNNencoder_gn(64, 32)
        self.convup3 = CNNencoder_gn(32, 32)
        self.concat4 = CNNencoder_gn(32, 16)
        self.convup4 = CNNencoder_gn(16, 16)
        self.concat5 = CNNencoder_ln(16, 3)
        self.convup5 = CNNencoder_ln(3, 3)
    def forward(self, x):
        v1 = self.upsample(x)
        u1 = self.concat1(v1)
        u1 = self.convup1(u1)
        u1 = self.upsample(u1)
        u2 = self.concat2(u1)
        u2 = self.convup2(u2)
        u2 = self.upsample(u2)
        u3 = self.concat3(u2)
        u3 = self.convup3(u3)
        u3 = self.upsample(u3)
        u4 = self.concat4(u3)
        u4 = self.convup4(u4)
        u4 = self.upsample(u4)
        u5 = self.concat5(u4)
        out = self.convup5(u5)
        return out






"""
Discriminator
"""

# class Cartoon_Discriminator(nn.Module):
#     def __init__(self, in_channels=3):
#         super().__init__()
#         self.pooling = nn.MaxPool2d(kernel_size=2)
#         self.conv1_1 = CNNencoder_gn(in_channels, 16)
#         self.conv1_2 = CNNencoder_gn_drop(16, 16)
#         self.conv2_1 = CNNencoder_gn(16, 16*2)
#         self.conv2_2 = CNNencoder_gn_drop(16*2, 16*2)
#         self.conv3_1 = CNNencoder_gn(16*2, 16*4)
#         self.conv3_2 = CNNencoder_gn_drop(16*4, 16*4)
#         self.conv4_1 = CNNencoder_gn(16*4, 16*8)
#         self.conv4_2 = CNNencoder_gn_drop(16*8, 16*8)
#         self.conv5_1 = CNNencoder_gn(16*8, 16*16)
#         self.conv5_2 = CNNencoder_gn_drop(16*16, 16*16)
#         self.realfake = nn.Linear(16*16*8*8, 1)
#         self.sig = nn.Sigmoid()
#     def forward(self, x):
#         c1 = self.conv1_1(x)
#         c1 = self.conv1_2(c1)
#         c2 = self.conv2_1(c1)
#         c2 = self.conv2_2(c2)
#         c3 = self.conv3_1(c2)
#         c3 = self.conv3_2(c3)
#         c4 = self.conv4_1(c3)
#         c4 = self.conv4_2(c4)
#         c5 = self.conv5_1(c4)
#         c5 = self.conv5_2(c5)
#         m = c5.view(-1, 16*16*8*8)
#         out = self.realfake(m)
#         return out
#
# class CelebA_Discriminator(nn.Module):
#     def __init__(self, in_channels=3):
#         super().__init__()
#         self.pooling = nn.MaxPool2d(kernel_size=2)
#         self.conv1_1 = CNNencoder_gn(in_channels, 16)
#         self.conv1_2 = CNNencoder_gn_drop(16, 16)
#         self.conv2_1 = CNNencoder_gn(16, 16*2)
#         self.conv2_2 = CNNencoder_gn_drop(16*2, 16*2)
#         self.conv3_1 = CNNencoder_gn(16*2, 16*4)
#         self.conv3_2 = CNNencoder_gn_drop(16*4, 16*4)
#         self.conv4_1 = CNNencoder_gn(16*4, 16*8)
#         self.conv4_2 = CNNencoder_gn_drop(16*8, 16*8)
#         self.conv5_1 = CNNencoder_gn(16*8, 16*16)
#         self.conv5_2 = CNNencoder_gn_drop(16*16, 16*16)
#         self.realfake = nn.Linear(16*16*8*8, 1)
#         self.sig = nn.Sigmoid()
#     def forward(self, x):
#         c1 = self.conv1_1(x)
#         c1 = self.conv1_2(c1)
#         c2 = self.conv2_1(c1)
#         c2 = self.conv2_2(c2)
#         c3 = self.conv3_1(c2)
#         c3 = self.conv3_2(c3)
#         c4 = self.conv4_1(c3)
#         c4 = self.conv4_2(c4)
#         c5 = self.conv5_1(c4)
#         c5 = self.conv5_2(c5)
#         m = c5.view(-1, 16*16*8*8)
#         out = self.realfake(m)
#         return out


"""
New Discriminator
"""

class Cartoon_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.conv1_1 = CNNencoder_gn(3, 16)
        self.conv1_2 = CNNencoder_gn(16, 16)
        self.conv2_1 = CNNencoder_gn(16, 32)
        self.conv2_2 = CNNencoder_gn(32, 32)
        self.conv3_1 = CNNencoder_gn(32, 64)
        self.conv3_2 = CNNencoder_gn(64, 64)
        self.conv4_1 = CNNencoder_gn(64, 128)
        self.conv4_2 = CNNencoder_gn(128, 128)
        self.conv5_1 = CNNencoder_gn(128, 256)
        self.conv5_2 = CNNencoder_gn(256, 256)
        self.conv6_1 = CNNencoder_gn(256, 256)
        self.conv7_1 = CNNencoder_gn(256, 256)
        self.realfake = nn.Linear(256*4*4, 1)
    def forward(self, x):
        c1 = self.conv1_1(x)
        # (B, 16, 256, 256)
        c1 = self.conv1_2(c1)
        # (B, 16, 256, 256)
        p1 = self.pooling(c1)
        # (B, 16, 128, 128)
        c2 = self.conv2_1(p1)
        # (B, 32, 128, 128)
        c2 = self.conv2_2(c2)
        # (B, 32, 128, 128)
        p2 = self.pooling(c2)
        # (B, 32, 64, 64)
        c3 = self.conv3_1(p2)
        # (B, 64, 64, 64)
        c3 = self.conv3_2(c3)
        # (B, 64, 64, 64)
        p3 = self.pooling(c3)
        # (B, 64, 32, 32)
        c4 = self.conv4_1(p3)
        # (B, 128, 32, 32)
        c4 = self.conv4_2(c4)
        # (B, 128, 32, 32)
        p4 = self.pooling(c4)
        # (B, 128, 16, 16)
        c5 = self.conv5_1(p4)
        # (B, 256, 16, 16)
        p5 = self.pooling(c5)
        # (B, 256, 8, 8)
        c6 = self.conv6_1(p5)
        # (B, 256, 8, 8)
        p6 = self.pooling(c6)
        # (B, 256, 4, 4)
        c7 = self.conv7_1(p6)
        # (B, 256, 4, 4)
        m = c7.view(-1, 256 * 4 * 4)
        out = self.realfake(m)
        return out


class CelebA_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.conv1_1 = CNNencoder_gn(3, 16)
        self.conv1_2 = CNNencoder_gn(16, 16)
        self.conv2_1 = CNNencoder_gn(16, 32)
        self.conv2_2 = CNNencoder_gn(32, 32)
        self.conv3_1 = CNNencoder_gn(32, 64)
        self.conv3_2 = CNNencoder_gn(64, 64)
        self.conv4_1 = CNNencoder_gn(64, 128)
        self.conv4_2 = CNNencoder_gn(128, 128)
        self.conv5_1 = CNNencoder_gn(128, 256)
        self.conv5_2 = CNNencoder_gn(256, 256)
        self.conv6_1 = CNNencoder_gn(256, 256)
        self.conv7_1 = CNNencoder_gn(256, 256)
        self.realfake = nn.Linear(256*4*4, 1)
    def forward(self, x):
        c1 = self.conv1_1(x)
        # (B, 16, 256, 256)
        c1 = self.conv1_2(c1)
        # (B, 16, 256, 256)
        p1 = self.pooling(c1)
        # (B, 16, 128, 128)
        c2 = self.conv2_1(p1)
        # (B, 32, 128, 128)
        c2 = self.conv2_2(c2)
        # (B, 32, 128, 128)
        p2 = self.pooling(c2)
        # (B, 32, 64, 64)
        c3 = self.conv3_1(p2)
        # (B, 64, 64, 64)
        c3 = self.conv3_2(c3)
        # (B, 64, 64, 64)
        p3 = self.pooling(c3)
        # (B, 64, 32, 32)
        c4 = self.conv4_1(p3)
        # (B, 128, 32, 32)
        c4 = self.conv4_2(c4)
        # (B, 128, 32, 32)
        p4 = self.pooling(c4)
        # (B, 128, 16, 16)
        c5 = self.conv5_1(p4)
        # (B, 256, 16, 16)
        p5 = self.pooling(c5)
        # (B, 256, 8, 8)
        c6 = self.conv6_1(p5)
        # (B, 256, 8, 8)
        p6 = self.pooling(c6)
        # (B, 256, 4, 4)
        c7 = self.conv7_1(p6)
        # (B, 256, 4, 4)
        m = c7.view(-1, 256 * 4 * 4)
        out = self.realfake(m)
        return out