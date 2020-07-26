import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.transforms as transforms
from ClassifierModels import Resnet18, Resnet18_V2
from AttentionUnetModel import AttentionUnet2D
from collections import OrderedDict
import matplotlib.pyplot as plt


class Relu1(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # print("fwd:", input[0])
        return input.clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] *= 0.0
        grad_input[input > 1] *= 0.0

        return grad_input


class BasicAutoEncoder(nn.Module):

    def __init__(self):
        super(BasicAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # 1x896x896
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=4, padding=2),
            nn.ELU(),
            # 1X224X224
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0),
            # 1x224x224
        )

        self.decoder = nn.Sequential(
            # 1x224x224
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(4),
            # 1x896x896
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        encoder_output = Relu1.apply(encoder_output)

        decoder_output = self.decoder(encoder_output)
        decoder_output = Relu1.apply(decoder_output)

        return encoder_output, decoder_output


class ImprovedAutoEncoder(nn.Module):

    def __init__(self):
        super(ImprovedAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # 1x896x896
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            # 1X224X224
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0),
            # 1x224x224
        )

        self.decoder = nn.Sequential(
            # 1x224x224
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(4),
            # 1x896x896
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        encoder_output = torch.sigmoid(encoder_output)

        decoder_output = self.decoder(encoder_output)
        decoder_output = torch.sigmoid(decoder_output)

        return encoder_output, decoder_output


class ImprovedAutoEncoder2(nn.Module):

    def __init__(self):
        super(ImprovedAutoEncoder2, self).__init__()

        self.encoder = nn.Sequential(
            # 1x896x896
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            # 1X224X224
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0),
            # 1x224x224
        )

        self.decoder = nn.Sequential(
            # 1x224x224
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.PixelShuffle(2),
            # 1x896x896
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        encoder_output = torch.sigmoid(encoder_output)

        decoder_output = self.decoder(encoder_output)
        decoder_output = torch.sigmoid(decoder_output)

        return encoder_output, decoder_output


class AE_Resnet18(nn.Module):

    def __init__(self, num_classes, is_backbone_trained=True):
        super(AE_Resnet18, self).__init__()

        self.num_classes = num_classes
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.auto_encoder = BasicAutoEncoder()
        self.classifier = Resnet18(num_classes=self.num_classes, is_trained=is_backbone_trained)

    def forward(self, x):

        encoder_output, decoder_output = self.auto_encoder(x)

        bs, c, h, w = encoder_output.shape
        latent_x = torch.Tensor(bs, 3, h, w).cuda()

        for img_no in range(bs):
            latent_x[img_no] = encoder_output[img_no]
            latent_x[img_no] = self.normalize(latent_x[img_no])  # broadcasting 1 channel to 3 channels

        classifier_output = self.classifier(latent_x)

        return decoder_output, classifier_output


class IMPROVED_AE_Resnet18(nn.Module):

    def __init__(self, num_classes, is_backbone_trained=True):
        super(IMPROVED_AE_Resnet18, self).__init__()

        self.num_classes = num_classes
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.auto_encoder = ImprovedAutoEncoder()
        self.classifier = Resnet18_V2(num_classes=self.num_classes, is_trained=is_backbone_trained)

    def forward(self, x):

        encoder_output, decoder_output = self.auto_encoder(x)

        bs, c, h, w = encoder_output.shape
        latent_x = torch.Tensor(bs, 3, h, w).cuda()

        for img_no in range(bs):
            latent_x[img_no] = encoder_output[img_no]
            latent_x[img_no] = self.normalize(latent_x[img_no])  # broadcasting 1 channel to 3 channels

        classifier_output = self.classifier(latent_x)

        return decoder_output, classifier_output


class AttentionUnetResnet18(nn.Module):

    def __init__(self, num_classes, is_backbone_trained=True):
        super(AttentionUnetResnet18, self).__init__()

        self.num_classes = num_classes
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.auto_encoder = AttentionUnet2D()
        self.classifier = Resnet18(num_classes=self.num_classes, is_trained=is_backbone_trained)

    def forward(self, x):

        encoder_output, decoder_output = self.auto_encoder(x)
        # encoder_output = Relu1.apply(encoder_output)
        encoder_output = torch.sigmoid(encoder_output)

        bs, c, h, w = encoder_output.shape
        latent_x = torch.Tensor(bs, 3, h, w).cuda()

        for img_no in range(bs):
            latent_x[img_no] = encoder_output[img_no]
            latent_x[img_no] = self.normalize(latent_x[img_no])  # broadcasting 1 channel to 3 channels

        classifier_output = self.classifier(latent_x)

        return decoder_output, classifier_output

