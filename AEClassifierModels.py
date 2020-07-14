import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.transforms as transforms
from ClassifierModels import Resnet18


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

        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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

        decoder_output = self.decoder(y)
        decoder_output = Relu1.apply(decoder_output)

        bs, c, h, w = encoder_output.shape
        y2 = torch.Tensor(bs, 3, h, w).cuda()

        for img_no in range(bs):
            y2[img_no] = encoder_output[img_no]
            y2[img_no] = self.normalize(y2[img_no])  # broadcasting 1 channel to 3 channels

        return encoder_output, decoder_output


class AE_Resnet18(nn.Module):

    def __init__(self, num_classes, is_backbone_trained=True,
                 pre_trained_classifier_path=None, pre_trained_ae_path=None):
        super(AE_Resnet18, self).__init__()

        self.auto_encoder = BasicAutoEncoder()

        self.classifier = Resnet18(num_classes=self.classCount, is_trained=is_backbone_trained)

        if pre_trained_ae_path is not None:
            self.auto_encoder.load_state_dict(torch.load(pre_trained_ae_path))

        if pre_trained_classifier_path is not None:
            self.classifier.load_state_dict(torch.load(pre_trained_classifier_path))

    def forward(self, x):

        encoder_output, decoder_output = self.auto_encoder(x)
        classifier_output = self.classifier(encoder_output)

        return decoder_output, classifier_output
