import os
import numpy as np
import time
import sys
import torch
import gc
from ModelTrainer import ModelTrainer

RESNET18 = 'RES-NET-18'
BASIC_AE = 'BASIC_AE'
AE_RESNET18 = 'AE-RES-NET-18'
ATTENTION_AE = 'ATTENTION_AE'
ATTENTION_AE_RESNET18 = 'IMPROVED-AE-RES-NET-18'
PATH_IMG_DIR = r'.\database'
PATH_FILE_TRAIN = r".\Dataset_files\train_1.txt"
PATH_FILE_VALIDATION = r".\Dataset_files\val_1.txt"
PATH_FILE_TEST = r'.\Dataset_files\test_1.txt'

def main():
    
    # run_test()
    run_train()
  

def run_train():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    if device == torch.device("cuda:0"):
        gc.collect()
        torch.cuda.empty_cache()
        print('Using GPU')
    else:
        print('Using CPU')

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    launch_timestamp = timestampDate + '-' + timestampTime
    
    # ---- Path to the directory with images

    # ---- Paths to the files with training, validation and testing sets.
    # ---- Each file should contains pairs [path to image, output vector]
    # ---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0

    
    # ---- Neural network parameters: type of the network, is it pre-trained
    # ---- on imagenet, number of classes
    # choose from: RESNET18, BASIC_AE, AE_RESNET18, ATTENTION_AE, ATTENTION_AE_RESNET18
    architecture_type = ATTENTION_AE
    is_backbone_pretrained = True
    num_classes = 14
    
    # ---- Training settings: batch size, maximum number of epochs
    batch_size = 64
    max_epoch = 100
    
    # ---- Parameters related to image transforms: size of the down-scaled image, cropped image
    trans_resize_size = None
    trans_crop_size = None
    trans_rotation_angle = None
    num_of_input_channels = 1
    # parameters per architecture:
    if architecture_type == RESNET18:
        # resize to 256 -> random crop to 224 -> random rotate [-5,5]
        trans_resize_size = 256
        trans_crop_size = 224
        trans_rotation_angle = 5
        num_of_input_channels = 3
    elif architecture_type == BASIC_AE or architecture_type == ATTENTION_AE:
        # random crop to 128
        trans_crop_size = 128
    elif architecture_type == AE_RESNET18 or architecture_type == ATTENTION_AE_RESNET18:
        # random crop to 896 -> random rotate [-5,5]
        trans_crop_size = 896
        trans_rotation_angle = 5

    path_saved_model = 'm-' + architecture_type + '-' + launch_timestamp + '.pth.tar'
    checkpoint = None # r"F:\AE_Classifier\m-RES-NET-18-18072020-150720.pth.tar"

    print ('Training NN architecture = ', architecture_type)
    model_trainer = ModelTrainer(architecture_type, num_of_input_channels, is_backbone_pretrained, num_classes, device)
    model_trainer.train(PATH_IMG_DIR, PATH_FILE_TRAIN, PATH_FILE_VALIDATION, batch_size,
                        max_epoch, trans_resize_size, trans_crop_size, trans_rotation_angle, launch_timestamp, checkpoint)
    
    print ('Testing the trained model')
    model_trainer.test(PATH_IMG_DIR, PATH_FILE_TEST, path_saved_model,
                       batch_size, trans_resize_size, trans_crop_size, None)


def run_test():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device == torch.device("cuda:0"):
        gc.collect()
        torch.cuda.empty_cache()

    architecture_type = RESNET18  # select from: RESNET18, AE_RESNET18, IMPROVED_AE_RESNET18
    is_backbone_pretrained = True
    num_classes = 14
    batch_size = 64
    trans_resize_size = None
    trans_crop_size = 896
    trans_rotation_angle = None
    num_of_input_channels = 3

    path_trained_model = r"F:\AE_Classifier\m-RES-NET-18-18072020-150720.pth.tar"

    model_trainer = ModelTrainer(architecture_type, num_of_input_channels, is_backbone_pretrained, num_classes, device)
    model_trainer.test(PATH_IMG_DIR, PATH_FILE_TEST, path_trained_model,
                       batch_size, trans_resize_size, trans_crop_size, trans_rotation_angle)


if __name__ == '__main__':
    main()






