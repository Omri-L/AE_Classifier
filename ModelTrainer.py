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
from AEClassifierModels import AE_Resnet18
from DatasetGenerator import DatasetGenerator
import PIL


def data_augmentations(resize_target, crop_target, normalization_vec, rotation_angle=None):

    transformList = []
    # optional augmentation:
    if resize_target is not None:
        transformList.append(transforms.Resize(resize_target))

    # basic augmentations:
    transformList.append(transforms.RandomCrop(crop_target))
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
    def __init__(self, architecture_type, is_backbone_trained, num_classes, device):
        self.architecture_type = architecture_type
        self.device = device
        self.num_classes = num_classes

        # -------------------- SETTINGS: NETWORK ARCHITECTURE
        if self.architecture_type == 'RES-NET-18':
            self.model = Resnet18(self.num_classes, is_backbone_trained).to(self.device)
        elif self.architecture_type == 'AE_RES-NET-18':
            self.model = AE_Resnet18(self.num_classes, is_backbone_trained, None, None).to(self.device)
            return
        elif self.architecture_type == 'IMPROVED_AE_RES-NET-18':
            print(self.architecture_type + " not supported yet")
            # self.model = IMPROVED_AE_Resnet18(num_classes, is_backbone_trained).to(self.device)
            return

        # model = torch.nn.DataParallel(model).to(device)

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
            out_auroc.append(roc_auc_score(np_gt_data[:, i], np_prediction[:]))

        return out_auroc

    def epoch_train(self, epoch_id, model, data_loader, optimizer, bce_loss, mse_loss):
        model.train()

        loss_value_mean = 0

        for batch_id, (input_img, target_label) in enumerate(data_loader):
            target_label = target_label.to(self.device, non_blocking=True)

            varInput = torch.autograd.Variable(input_img).to(self.device)
            varTarget = torch.autograd.Variable(target_label).to(self.device)
            varOutput = model(varInput)

            loss_value = bce_loss(varOutput, varTarget)  # TODO: use smarter way to use the loss according to architecture

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_value_mean += loss_value

            if batch_id % (int(len(data_loader)*0.2)) == 0:
                print("----> EpochID: {}, BatchID/NumBatches: {}/{}, mean train loss: {}"
                      .format(epoch_id + 1, batch_id + 1, len(data_loader), loss_value_mean / (batch_id + 1)))

            if self.device == torch.device("cuda:0"):
                gc.collect()
                torch.cuda.empty_cache()

        loss_value_mean /= len(data_loader)
        print("-------> EpochID: {}, final mean train loss: {}".format(epoch_id + 1, loss_value_mean))

    def epoch_validation(self, model, data_loader, bce_loss, mse_loss):
        model.eval()
        loss_val = 0
        loss_val_norm = 0
        loss_tensor_mean = 0

        for i, (input_img, target_label) in enumerate(data_loader):
            target_label = target_label.to(self.device, non_blocking=True)

            torch.no_grad()
            varInput = torch.autograd.Variable(input_img).to(self.device)
            varTarget = torch.autograd.Variable(target_label).to(self.device)
            varOutput = model(varInput)

            # TODO: check all of the following
            loss_tensor = bce_loss(varOutput, varTarget)  # TODO: use smarter way to use the loss according to architecture
            loss_tensor_mean += loss_tensor

            loss_val += loss_tensor.item()
            loss_val_norm += 1

            if self.device == torch.device("cuda:0"):
                gc.collect()
                torch.cuda.empty_cache()

        out_loss = loss_val / loss_val_norm
        loss_tensor_mean = loss_tensor_mean / loss_val_norm
        return out_loss, loss_tensor_mean

    def train(self, path_img_dir, path_file_train, path_file_validation, batch_size,
              max_epochs, trans_resize_size, trans_crop_size, trans_rotation_angle, launch_timestamp, checkpoint):

        # -------------------- SETTINGS: DATA AUGMENTATION
        normalization_vec = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformSequence = data_augmentations(trans_resize_size, trans_crop_size,
                                               normalization_vec, trans_rotation_angle)

        # -------------------- SETTINGS: DATASET BUILDERS
        dataset_train = DatasetGenerator(pathImageDirectory=path_img_dir, pathDatasetFile=path_file_train,
                                         transform=transformSequence)
        dataset_validation = DatasetGenerator(pathImageDirectory=path_img_dir, pathDatasetFile=path_file_validation,
                                              transform=transformSequence)
              
        dataLoader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                      shuffle=True, num_workers=0, pin_memory=True)
        dataLoader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,
                                           shuffle=False, num_workers=0, pin_memory=True)
        
        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER  # TODO: add parameters of the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode ='min') # TODO: check what it does?
                
        # -------------------- SETTINGS: LOSS
        bce_loss = torch.nn.BCELoss(reduction='mean')
        mse_loss = torch.nn.MSELoss(reduction='mean')

        # ---- Load checkpoint
        if checkpoint is not None:
            modelCheckpoint = torch.load(checkpoint)
            self.model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        # ---- TRAIN THE NETWORK
        min_loss = 100000
        
        for epoch_id in range(0, max_epochs):
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime

            self.epoch_train(epoch_id, self.model, dataLoader_train, optimizer, bce_loss, mse_loss)
            loss_val, loss_val_tensor = self.epoch_validation(self.model, dataLoader_validation, bce_loss, mse_loss)
            print("-------> EpochID: {}, mean validation loss: {}".format(epoch_id + 1, loss_val))

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(loss_val_tensor.item())
            
            if loss_val < min_loss:
                min_loss = loss_val
                torch.save({'model_type': self.architecture_type,
                            'epoch': epoch_id + 1,
                            'state_dict': self.model.state_dict(),
                            'best_loss': min_loss,
                            'optimizer' : optimizer.state_dict()},
                           'm-' + launch_timestamp + '.pth.tar')
                print('Epoch [' + str(epoch_id + 1) + '] [save] [' + timestampEND + '] loss= ' + str(loss_val))
            else:
                print('Epoch [' + str(epoch_id + 1) + '] [----] [' + timestampEND + '] loss= ' + str(loss_val))


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
    
    def test(self, path_img_dir, path_file_test, path_trained_model,
             batch_size, trans_resize_size, trans_crop_size, trans_rotation_angle):

        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
                        'Hernia', 'Healthy']
        
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
        normalization_vec = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformSequence = data_augmentations(trans_resize_size, trans_crop_size,
                                               normalization_vec, trans_rotation_angle)
        
        dataset_test = DatasetGenerator(pathImageDirectory=path_img_dir, pathDatasetFile=path_file_test,
                                        transform=transformSequence)
        data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=0,
                                      shuffle=False, pin_memory=True)
        
        out_gt = torch.FloatTensor().to(self.device)
        out_pred = torch.FloatTensor().to(self.device)
       
        self.model.eval()
        for i, (input_img, target) in enumerate(data_loader_test):
            
            target = target.to(self.device)
            out_gt = torch.cat((out_gt, target), 0)
            
            bs, c, h, w = input_img.size()

            torch.no_grad()
            varInput = torch.autograd.Variable(input_img.view(-1, c, h, w).to(self.device))
            
            out = self.model(varInput)
            out_mean = out.view(bs, -1).mean(1)
            
            out_pred = torch.cat((out_pred, out_mean.data), 0)

        auroc_individual = self.compute_AUROC(out_gt, out_pred)
        auroc_mean = np.array(auroc_individual).mean()
        
        print ('AUROC mean ', auroc_mean)
        
        for i in range (0, len(auroc_individual)):
            print(CLASS_NAMES[i], ' ', auroc_individual[i])

        return





