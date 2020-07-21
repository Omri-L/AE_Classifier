import os
import numpy as np
import time
import sys
import torch
import gc
from ModelTrainer import ModelTrainer

# defines - do not set
RESNET18 = 'RES-NET-18'
BASIC_AE = 'BASIC_AE'
AE_RESNET18 = 'AE-RES-NET-18'
ATTENTION_AE = 'ATTENTION_AE'
ATTENTION_AE_RESNET18 = 'IMPROVED-AE-RES-NET-18'

# Configuration parameters for set:
###################################
ARCHITECTURE_TYPE = RESNET18  # select from: RESNET18, BASIC_AE, AE_RESNET18, ATTENTION_AE, ATTENTION_AE_RESNET18
RUN_MODE = 'Train_Test'  # select from: 'Train_Test', 'Train_only', 'Test_only'
# Set paths:
PATH_IMG_DIR = r'.\database'
PATH_FILE_TRAIN = r'.\Dataset_files\train_1.txt'
PATH_FILE_VALIDATION = r'.\Dataset_files\val_1.txt'
PATH_FILE_TEST = r'.\Dataset_files\test_1.txt'
SAVE_DIR_NAME = 'TrainedModels'
SAVE_PRINTS_TO_FILE = True  # only for train modes
CHECKPOINT_FOR_TRAINING = None  # r'./m-RES-NET-18-20072020-073848.pth.tar'  # None # For training continuation
TRAINED_CLASSIFIER_MODEL = None  # set trained classifier model path only for the joint training
TRAINED_AE_MODEL = None  # set trained AE model path only for the joint training
TRAINED_MODEL_FOR_TEST = r'./m-RES-NET-18-19072020-151816.pth.tar'
# Hyper parameters:
BATCH_SIZE = 64
MAX_EPOCH = 2
OPTIMIZER_LR = 0.0001
OPTIMIZER_WEIGHT_DECAY = 5e-4
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 5
LOSS_LAMBDA = 0.9  # relevant only for the joint training


def main():

    print('Running {} mode'.format(RUN_MODE))
    print('Settings:\n' +
          'PATH_IMG_DIR {}\n'.format(PATH_IMG_DIR) +
          'PATH_FILE_TRAIN {}\n'.format(PATH_FILE_TRAIN) +
          'PATH_FILE_VALIDATION {}\n'.format(PATH_FILE_VALIDATION) +
          'PATH_FILE_TEST {}\n'.format(PATH_FILE_TEST) +
          'SAVE_DIR {}\n'.format(SAVE_DIR_NAME) +
          'CHECKPOINT_FOR_TRAINING {}\n'.format(CHECKPOINT_FOR_TRAINING) +
          'TRAINED_CLASSIFIER_MODEL {}\n'.format(TRAINED_CLASSIFIER_MODEL) +
          'TRAINED_AE_MODEL {}\n'.format(TRAINED_AE_MODEL) +
          'TRAINED_MODEL_FOR_TEST {}\n'.format(TRAINED_MODEL_FOR_TEST) +
          'BATCH_SIZE {}\n'.format(BATCH_SIZE) +
          'MAX_EPOCH {}\n'.format(MAX_EPOCH) +
          'OPTIMIZER_LR {}\n'.format(OPTIMIZER_LR) +
          'OPTIMIZER_WEIGHT_DECAY {}\n'.format(OPTIMIZER_WEIGHT_DECAY) +
          'SCHEDULER_FACTOR {}\n'.format(SCHEDULER_FACTOR) +
          'SCHEDULER_PATIENCE {}\n'.format(SCHEDULER_PATIENCE) +
          'LOSS_LAMBDA {}\n'.format(LOSS_LAMBDA)
          )

    if os.path.exists(SAVE_DIR_NAME) is False:
        os.mkdir(SAVE_DIR_NAME)

    if RUN_MODE == 'Test_only':
        run_test()
    else:
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

    # ---- Neural network parameters: type of the network, is it pre-trained
    # ---- on imagenet, number of classes
    # choose from: RESNET18, BASIC_AE, AE_RESNET18, ATTENTION_AE, ATTENTION_AE_RESNET18
    architecture_type = ARCHITECTURE_TYPE
    is_backbone_pretrained = True
    num_classes = 14
    
    # ---- Training settings: batch size, maximum number of epochs
    batch_size = BATCH_SIZE
    max_epoch = MAX_EPOCH

    # ---- Optimizer settings:
    optimizer_lr = OPTIMIZER_LR
    optimizer_weight_decay = OPTIMIZER_WEIGHT_DECAY
    scheduler_factor = SCHEDULER_FACTOR
    scheduler_patience = SCHEDULER_PATIENCE
    lambda_loss = LOSS_LAMBDA
    
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

    sub_dir = os.getcwd() + '\\' + SAVE_DIR_NAME + '\\' + architecture_type + launch_timestamp
    os.mkdir(sub_dir)
    if SAVE_PRINTS_TO_FILE:
        log_path = sub_dir + "\\log.txt"

    checkpoint_training = CHECKPOINT_FOR_TRAINING

    print('Training NN architecture = ', architecture_type)
    model_trainer = ModelTrainer(architecture_type, num_of_input_channels, is_backbone_pretrained, num_classes,
                                 lambda_loss, device,
                                 sub_dir, log_path, TRAINED_CLASSIFIER_MODEL, TRAINED_AE_MODEL)
    model_trainer.train(PATH_IMG_DIR, PATH_FILE_TRAIN, PATH_FILE_VALIDATION, batch_size, max_epoch,
                        optimizer_lr, optimizer_weight_decay, scheduler_factor, scheduler_patience,
                        trans_resize_size, trans_crop_size, trans_rotation_angle,
                        launch_timestamp, checkpoint_training)
    
    if RUN_MODE == 'Train_Test':
        print('Testing the trained model')
        saved_model_name = sub_dir + '\\m-' + architecture_type + '-' + launch_timestamp + '.pth.tar'
        auroc_mean = model_trainer.test(PATH_IMG_DIR, PATH_FILE_TEST, saved_model_name,
                           batch_size, trans_resize_size, trans_crop_size)


def run_test():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device == torch.device("cuda:0"):
        gc.collect()
        torch.cuda.empty_cache()
        print('Using GPU')
    else:
        print('Using CPU')

    architecture_type = ARCHITECTURE_TYPE  # select from: RESNET18, AE_RESNET18, IMPROVED_AE_RESNET18
    is_backbone_pretrained = True
    num_classes = 14
    batch_size = BATCH_SIZE

    trans_resize_size = None
    trans_crop_size = None
    num_of_input_channels = 1
    # parameters per architecture:
    if architecture_type == RESNET18:
        # resize to 256 -> crop to 224
        trans_resize_size = 256
        trans_crop_size = 224
        num_of_input_channels = 3
    elif architecture_type == BASIC_AE or architecture_type == ATTENTION_AE:
        # crop to 128
        trans_crop_size = 128
    elif architecture_type == AE_RESNET18 or architecture_type == ATTENTION_AE_RESNET18:
        # crop to 896
        trans_crop_size = 896

    path_trained_model = TRAINED_MODEL_FOR_TEST
    if os.path.isfile(path_trained_model) is False:
        print('TRAINED_MODEL_FOR_TEST is not exists! please provide valid trained model path')

    print('Testing NN architecture = ', architecture_type)
    model_trainer = ModelTrainer(architecture_type, num_of_input_channels, is_backbone_pretrained, num_classes,
                                 LOSS_LAMBDA, device)
    auroc_mean = model_trainer.test(PATH_IMG_DIR, PATH_FILE_TEST, path_trained_model,
                       batch_size, trans_resize_size, trans_crop_size)


if __name__ == '__main__':
    main()






