import os
import numpy as np
import time
import sys
import torch
import gc
from ModelTrainerFromAugmentedData import ModelTrainer

RESNET18 = 'RES-NET-18'
BASIC_AE = 'BASIC_AE'
AE_RESNET18 = 'AE-RES-NET-18'
ATTENTION_AE = 'ATTENTION_AE'
ATTENTION_AE_RESNET18 = 'IMPROVED-AE-RES-NET-18'


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
    
    # ---- Path to the directory with data
    path_file_train = r"E:\AE_Classifier\database\Augmented_data_RES-NET-18\train"
    path_file_validation = r"E:\AE_Classifier\database\Augmented_data_RES-NET-18\val"
    path_file_test = r"E:\AE_Classifier\database\Augmented_data_RES-NET-18\test"

    # ---- Neural network parameters: type of the network, is it pre-trained
    # ---- on imagenet, number of classes
    # choose from: RESNET18, BASIC_AE, AE_RESNET18, ATTENTION_AE, ATTENTION_AE_RESNET18
    architecture_type = RESNET18
    is_backbone_pretrained = True
    num_classes = 14
    
    # ---- Training settings: batch size, maximum number of epochs
    batch_size = 32
    max_epoch = 2
    
    num_of_input_channels = 1
    # parameters per architecture:
    if architecture_type == RESNET18:
        num_of_input_channels = 3

    path_saved_model = 'm-' + architecture_type + '-' + launch_timestamp + '.pth.tar'
    print ('Training NN architecture = ', architecture_type)
    model_trainer = ModelTrainer(architecture_type, num_of_input_channels, is_backbone_pretrained, num_classes, device)
    model_trainer.train(path_file_train, path_file_validation, batch_size, max_epoch, launch_timestamp, None)
    model_trainer.test(path_file_test, path_saved_model, batch_size)


def run_test():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    if device == torch.device("cuda:0"):
        gc.collect()
        torch.cuda.empty_cache()
        print('Using GPU')
    else:
        print('Using CPU')

    path_file_test = r"E:\AE_Classifier\database\Augmented_data_RES-NET-18\test"
    path_trained_model = None
    architecture_type = RESNET18  # select from: RESNET18, AE_RESNET18, IMPROVED_AE_RESNET18
    is_backbone_pretrained = True
    num_classes = 14
    batch_size = 16
    num_of_input_channels = 3

    model_trainer = ModelTrainer(architecture_type, num_of_input_channels, is_backbone_pretrained, num_classes, device)
    model_trainer.test(path_file_test, path_trained_model, batch_size)


if __name__ == '__main__':
    main()






