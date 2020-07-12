import os
import numpy as np
import time
import sys
import torch
import gc
from ModelTrainer import ModelTrainer

RESNET18 = 'RES-NET-18'
AE_RESNET18 = 'AE-RES-NET-18'
IMPROVED_AE_RESNET18 = 'IMPROVED-AE-RES-NET-18'


def main():
    
    # run_test()
    run_train()
  

def run_train():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda:0"):
        gc.collect()
        torch.cuda.empty_cache()

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    launch_timestamp = timestampDate + '-' + timestampTime
    
    # ---- Path to the directory with images
    path_img_dir = r'D:\DL_MI_project\ChestXRay14\images'
    
    # ---- Paths to the files with training, validation and testing sets.
    # ---- Each file should contains pairs [path to image, output vector]
    # ---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    path_file_train = r'D:\DL_MI_project\ChestXRay14\train_only_15_small_for_check.txt'
    path_file_validation = r'D:\DL_MI_project\ChestXRay14\val_only_15_small_for_check.txt'
    path_file_test = r'D:\DL_MI_project\ChestXRay14\test_final_15_small_for_check.txt'
    
    # ---- Neural network parameters: type of the network, is it pre-trained
    # ---- on imagenet, number of classes
    architecture_type = RESNET18
    is_backbone_pretrained = True
    num_classes = 15
    
    # ---- Training settings: batch size, maximum number of epochs
    batch_size = 4
    max_epoch = 3
    
    # ---- Parameters related to image transforms: size of the down-scaled image, cropped image
    trans_resize_size = 256
    trans_crop_size = 224
    trans_rotation_angle = 5
        
    path_saved_model = 'm-' + launch_timestamp + '.pth.tar'
    
    print ('Training NN architecture = ', architecture_type)
    model_trainer = ModelTrainer(architecture_type, is_backbone_pretrained, num_classes, device)
    model_trainer.train(path_img_dir, path_file_train, path_file_validation, batch_size, max_epoch,
                        trans_resize_size, trans_crop_size, trans_rotation_angle, launch_timestamp, None)
    
    print ('Testing the trained model')
    model_trainer.test(path_img_dir, path_file_test, None, num_classes,
                       batch_size, trans_resize_size, trans_crop_size, trans_rotation_angle,
                       launch_timestamp)


def run_test():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device == torch.device("cuda:0"):
        gc.collect()
        torch.cuda.empty_cache()

    path_img_dir = r'D:\DL_MI_project\ChestXRay14\images'
    path_file_test = r'D:\DL_MI_project\ChestXRay14\test_final_14.txt'
    architecture_type = RESNET18  # select from: RESNET18, AE_RESNET18, IMPROVED_AE_RESNET18
    is_backbone_pretrained = True
    num_classes = 15
    batch_size = 4
    trans_resize_size = 256
    trans_crop_size = 224
    trans_rotation_angle = 5
    
    path_trained_model = ''
    
    launch_timestamp = ''

    model_trainer = ModelTrainer(architecture_type, is_backbone_pretrained, num_classes, device)
    model_trainer.test(path_img_dir, path_file_test, path_trained_model, num_classes,
                       batch_size, trans_resize_size, trans_crop_size, trans_rotation_angle,
                       launch_timestamp)


if __name__ == '__main__':
    main()






