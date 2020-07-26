import numpy as np
import time
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from DatasetGenerator import DatasetGenerator
import PIL
import matplotlib.pyplot as plt
import os
import gc
import xlsxwriter

RESNET18 = 'RES-NET-18'
BASIC_AE = 'BASIC_AE'
IMPROVED_AE = 'IMPROVED_AE'
AE_RESNET18 = 'AE-RES-NET-18'
ATTENTION_AE = 'ATTENTION_AE'
IMPROVED_AE_RESNET18 = 'IMPROVED-AE-RES-NET-18'
ATTENTION_AE_RESNET18 = 'ATTENTION-AE-RES-NET-18'
COMBINED_ARCH = [AE_RESNET18, IMPROVED_AE_RESNET18, ATTENTION_AE_RESNET18]
AE_ARCH = [BASIC_AE, IMPROVED_AE, ATTENTION_AE]
CLASSIFIER_ARCH = [RESNET18]
DATA_PARALLEL = False
NUM_CLASSES = 14

PATH_IMG_DIR = r'D:\DL_MI_project\ChestXRay14\images'
PATH_FILE_TRAIN = r"D:\DL_MI_project\ChestXRay14\original_split\train_1.txt"
PATH_FILE_VALIDATION = r"D:\DL_MI_project\ChestXRay14\original_split\val_1.txt"
PATH_FILE_TEST = r'D:\DL_MI_project\ChestXRay14\original_split\test_1.txt'
# PATH_FILE_TRAIN = r"D:\DL_MI_project\ChestXRay14\train_only_14_small_for_check.txt"
# PATH_FILE_VALIDATION = r"D:\DL_MI_project\ChestXRay14\val_only_14_small_for_check.txt"
# PATH_FILE_TEST = r"D:\DL_MI_project\ChestXRay14\test_final_14_small_for_check.txt"
# PATH_FILE_TRAIN = r".\Dataset_files\train_one.txt"
# PATH_FILE_VALIDATION = r".\Dataset_files\val_one.txt"
# PATH_FILE_TEST = r'.\Dataset_files\test_one.txt'
