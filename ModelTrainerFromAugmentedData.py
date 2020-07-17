import numpy as np
import time

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc

from sklearn.metrics.ranking import roc_auc_score

from ClassifierModels import Resnet18
from AEClassifierModels import BasicAutoEncoder, AE_Resnet18
from DatasetGeneratorTensors import DatasetGeneratorAugTensors
import PIL
import matplotlib.pyplot as plt
import glob
import random


def data_augmentations(resize_target, crop_target, normalization_vec, rotation_angle=None):

    transformList = []
    # optional augmentation:
    if resize_target is not None:
        transformList.append(transforms.Resize(resize_target))

    # basic augmentations:
    transformList.append(transforms.CenterCrop(crop_target))
    transformList.append(transforms.RandomHorizontalFlip())

    # optional augmentation
    if rotation_angle is not None:
        transformList.append(transforms.RandomRotation(rotation_angle, PIL.Image.BILINEAR))

    # basic augmentations (end)
    transformList.append(transforms.ToTensor())
    if normalization_vec is not None:
        transformList.append(transforms.Normalize(normalization_vec[0], normalization_vec[1]))

    transformSequence = transforms.Compose(transformList)
    return transformSequence


def plt_data(data_train, data_val, titleStr, save_fig=False, save_dir=''):
    fig1 = plt.figure()
    plt.xlabel('Epoch #')
    plt.plot(data_train, color='b', label='Train')
    titleStr_train = titleStr + '_Train'
    plt.title(titleStr_train)
    plt.legend(loc='upper right')
    if save_fig:
        fig1.savefig(save_dir + titleStr_train + '.png', dpi=100)

    fig2 = plt.figure()
    plt.xlabel('Epoch #')
    plt.plot(data_val, color='r', label='Validation')
    titleStr_test = titleStr + '_Validation'
    plt.title(titleStr_test)
    plt.legend(loc='upper right')
    if save_fig:
        fig2.savefig(save_dir + titleStr_test + '.png', dpi=100)

    plt.close('all')


class ModelTrainer:
    # ---- Train the densenet network
    # ---- pathDirData - path to the directory that contains images
    # ---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    # ---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    # ---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    # ---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    # ---- nnClassCount - number of output classes
    # ---- trBatchSize - batch size
    # ---- trMaxEpoch - number of epochs
    # ---- transResize - size of the image to scale down to (not used in current implementation)
    # ---- transCrop - size of the cropped image
    # ---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    # ---- checkpoint - if not None loads the model and continues training
    def __init__(self, architecture_type, num_of_input_channels, is_backbone_trained, num_classes, device):
        self.architecture_type = architecture_type
        self.device = device
        self.num_classes = num_classes
        self.num_of_input_channels = num_of_input_channels

        # -------------------- SETTINGS: NETWORK ARCHITECTURE
        if self.architecture_type == 'RES-NET-18':
            self.model = Resnet18(self.num_classes, is_backbone_trained).to(self.device)
        elif self.architecture_type == 'BASIC_AE':
            self.model = BasicAutoEncoder().to(self.device)
        elif self.architecture_type == 'AE-RES-NET-18':
            self.model = AE_Resnet18(self.num_classes, is_backbone_trained, None, None).to(self.device)
        # elif self.architecture_type == 'ATTENTION_AE_RES-NET-18':
        #     self.model = Attention_AE_Resnet18(num_classes, is_backbone_trained).to(self.device)

        # self.model = torch.nn.DataParallel(self.model).to(self.device)

        self.bce_loss = torch.nn.BCELoss(reduction='mean')
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def compute_AUROC(self, gt_data, prediction):
        """
        Computes area under ROC curve
        :param gt_data - ground truth data
        :param prediction - predicted data
        :return out_auroc - area under ROC curve vector (value for each class)
        """
        out_auroc = []

        np_gt_data = gt_data.cpu().numpy()
        np_prediction = prediction.cpu().numpy()

        for i in range(self.num_classes):
            if len(np.unique(np_gt_data[:, i])) != 2:
                out_auroc.append(-1)  # error, no data with that class!
            else:
                out_auroc.append(roc_auc_score(np_gt_data[:, i], np_prediction[:]))

        return out_auroc

    def epoch_train(self, epoch_id, optimizer, dataloader_train):
        self.model.train()

        loss_value_mean = 0
        for batch_id, (input_img, target_label) in enumerate(dataloader_train):
            input_img = input_img.to(self.device, non_blocking=True)
            target_label = target_label.to(self.device, non_blocking=True)

            varInput = torch.autograd.Variable(input_img).to(self.device)
            varTarget = torch.autograd.Variable(target_label).to(self.device)
            varOutput = self.model(varInput)

            # TODO: use smarter way to use the loss according to architecture
            if self.architecture_type == 'RES-NET-18':
                loss_value = self.bce_loss(varOutput, varTarget)
            elif self.architecture_type == 'BASIC_AE':
                encoder_output, decoder_output = varOutput
                loss_value = self.mse_loss(decoder_output, varInput)
            elif self.architecture_type == 'AE-RES-NET-18':
                decoder_output, classifier_output = varOutput
                lambda_loss = 0.9
                loss_bce_value = self.bce_loss(classifier_output, varTarget)
                loss_mse_value = self.mse_loss(decoder_output, varInput)
                loss_value = lambda_loss * loss_bce_value + (1-lambda_loss) * loss_mse_value

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_value_mean += loss_value.item()

            if batch_id % (int(len(dataloader_train)*0.2)) == 0:
                print("----> EpochID: {}, BatchID/NumBatches: {}/{}, mean train loss: {}"
                      .format(epoch_id + 1, batch_id + 1, len(dataloader_train), loss_value_mean / (batch_id + 1)))

            if self.device == torch.device("cuda:0"):
                gc.collect()
                torch.cuda.empty_cache()

        loss_value_mean /= len(dataloader_train)
        return loss_value_mean

    def epoch_validation(self, dataloader_validation):
        self.model.eval()
        loss_val = 0
        loss_val_norm = 0
        loss_tensor_mean = 0

        with torch.no_grad():
            for batch_id, (input_img, target_label) in enumerate(dataloader_validation):
                input_img = input_img.to(self.device, non_blocking=True)
                target_label = target_label.to(self.device, non_blocking=True)
                varInput = torch.autograd.Variable(input_img).to(self.device)
                varTarget = torch.autograd.Variable(target_label).to(self.device)
                varOutput = self.model(varInput)

                # TODO: use smarter way to use the loss according to architecture
                if self.architecture_type == 'RES-NET-18':
                    loss_value = self.bce_loss(varOutput, varTarget)
                elif self.architecture_type == 'BASIC_AE':
                    encoder_output, decoder_output = varOutput
                    loss_value = self.mse_loss(decoder_output, varInput)
                elif self.architecture_type == 'AE-RES-NET-18':
                    decoder_output, classifier_output = varOutput
                    lambda_loss = 0.9
                    loss_bce_value = self.bce_loss(classifier_output, varTarget)
                    loss_mse_value = self.mse_loss(decoder_output, varInput)
                    loss_value = lambda_loss * loss_bce_value + (1-lambda_loss) * loss_mse_value

                loss_tensor_mean += loss_value
                loss_val += loss_value.item()
                loss_val_norm += 1

                if self.device == torch.device("cuda:0"):
                    gc.collect()
                    torch.cuda.empty_cache()

        out_loss = loss_val / loss_val_norm
        loss_tensor_mean = loss_tensor_mean / loss_val_norm
        return out_loss, loss_tensor_mean

    def train(self, path_file_train, path_file_validation, batch_size, max_epochs, launch_timestamp, checkpoint):

        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER  # TODO: add parameters of the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode ='min')

        # -------------------- SETTINGS: DATASET BUILDERS
        dataset_train = DatasetGeneratorAugTensors(path_to_tensor_files=path_file_train)
        dataset_validation = DatasetGeneratorAugTensors(path_to_tensor_files=path_file_validation)

        dataLoader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                      shuffle=True, num_workers=0)
        dataLoader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,
                                           shuffle=False, num_workers=0)


        # ---- Load checkpoint
        if checkpoint is not None:
            modelCheckpoint = torch.load(checkpoint)
            self.model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        # ---- TRAIN THE NETWORK
        min_loss = 100000

        loss_train_list = []
        loss_validation_list = []

        for epoch_id in range(0, max_epochs):
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            loss_train = self.epoch_train(epoch_id, optimizer, dataLoader_train)
            loss_validation, loss_validation_tensor = self.epoch_validation(dataLoader_validation)
            print("-------> EpochID: {}, mean train loss: {}".format(epoch_id + 1, loss_train))
            print("-------> EpochID: {}, mean validation loss: {}".format(epoch_id + 1, loss_validation))
            loss_train_list.append(loss_train)
            loss_validation_list.append(loss_validation)

            if epoch_id % (round(max_epochs*0.1)+1) == 0:
                plt_data(loss_train_list, loss_validation_list, "Loss_train_vs_validation_" + self.architecture_type,
                         True, "")

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(loss_validation_tensor.item())
            
            if loss_validation < min_loss:
                min_loss = loss_validation
                torch.save({'model_type': self.architecture_type,
                            'epoch': epoch_id + 1,
                            'state_dict': self.model.state_dict(),
                            'best_loss': min_loss,
                            'optimizer' : optimizer.state_dict()},
                           'm-' + self.architecture_type + '-' + launch_timestamp + '.pth.tar')
                print('Epoch [' + str(epoch_id + 1) + '] [save] [' + timestampEND + '] loss= ' + str(loss_validation))
            else:
                print('Epoch [' + str(epoch_id + 1) + '] [----] [' + timestampEND + '] loss= ' + str(loss_validation))

        print("finish training!")

    # ---- Test the trained network
    # ---- pathDirData - path to the directory that contains images
    # ---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    # ---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    # ---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    # ---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    # ---- nnClassCount - number of output classes
    # ---- trBatchSize - batch size
    # ---- trMaxEpoch - number of epochs
    # ---- transResize - size of the image to scale down to (not used in current implementation)
    # ---- transCrop - size of the cropped image
    # ---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    # ---- checkpoint - if not None loads the model and continues training
    
    def test(self, path_file_test, path_trained_model, batch_size):

        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
                        'Hernia']
        
        cudnn.benchmark = True  # TODO check what is that?

        self.model.to(self.device)
        # model = torch.nn.DataParallel(model).cuda()
        # fix from: https://github.com/bearpaw/pytorch-classification/issues/27
        if path_trained_model is not None:
            modelCheckpoint = torch.load(path_trained_model)
            # model.load_state_dict(modelCheckpoint['state_dict'])

            state_dict = modelCheckpoint['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if 'module.' in k:
                    k = k.replace('module.densenet121', 'densenet121')
                if 'norm.1' or 'norm.2' in k:
                    k = k.replace('norm.1', 'norm1')
                    k = k.replace('norm.2', 'norm2')
                if 'conv.1' or 'conv.2' in k:
                    k = k.replace('conv.1', 'conv1')
                    k = k.replace('conv.2', 'conv2')
                new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict)

        # -------------------- SETTINGS: DATA AUGMENTATION
        dataset_test = DatasetGeneratorAugTensors(path_to_tensor_files=path_file_test)
        data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=0, shuffle=False)
        
        out_gt = torch.FloatTensor().to(self.device)
        out_pred = torch.FloatTensor().to(self.device)

        self.model.eval()
        with torch.no_grad():
            for batch_id, (input_img, target) in enumerate(data_loader_test):

                target = target.to(self.device)
                out_gt = torch.cat((out_gt, target), 0)

                bs, c, h, w = input_img.size()
                varInput = torch.autograd.Variable(input_img.view(-1, c, h, w).to(self.device))

                out = self.model(varInput)
                if self.architecture_type == 'BASIC_AE':
                    encoder_output, decoder_output = out
                elif self.architecture_type == 'AE-RES-NET-18':
                    decoder_output, classifier_output = out

                if self.architecture_type == 'RES-NET-18' or self.architecture_type == 'AE-RES-NET-18':
                    out_mean = out.view(bs, -1).mean(1)
                    out_pred = torch.cat((out_pred, out_mean.data), 0)

        if self.architecture_type != 'ATTENTION_AE' or self.architecture_type != 'BASIC_AE':
            auroc_individual = self.compute_AUROC(out_gt, out_pred)
            auroc_mean = np.array(auroc_individual).mean()
            print ('AUROC mean ', auroc_mean)
            for i in range (0, len(auroc_individual)):
                print(CLASS_NAMES[i], ' ', auroc_individual[i])

        return




