import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import gc
import os
from DatasetGenerator import DatasetGenerator
import PIL


def data_augmentations(resize_target, crop_target, normalization_vec, randomHorizontal=False, rotation_angle=None):

    transformList = []
    # optional augmentation:
    if resize_target is not None:
        transformList.append(transforms.Resize(resize_target))

    # basic augmentations:
    transformList.append(transforms.CenterCrop(crop_target))
    if randomHorizontal:
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


def create_dataset(path_img_dir, path_file_train, path_file_validation, batch_size,
                   trans_resize_size, trans_crop_size, trans_rotation_angle, normalization_vec,
                   save_data_train_path, save_data_val_path, device, num_of_input_channels):

    transformSequenceTrain = data_augmentations(trans_resize_size, trans_crop_size, normalization_vec,
                                                True, trans_rotation_angle)
    transformSequenceValidation = data_augmentations(trans_resize_size, trans_crop_size, normalization_vec,
                                                     False, None)

    # -------------------- SETTINGS: DATASET BUILDERS
    dataset_train = DatasetGenerator(pathImageDirectory=path_img_dir, pathDatasetFile=path_file_train,
                                        transform=transformSequenceTrain, num_img_chs=num_of_input_channels)
    dataset_validation = DatasetGenerator(pathImageDirectory=path_img_dir, pathDatasetFile=path_file_validation,
                                            transform=transformSequenceValidation, num_img_chs=num_of_input_channels)

    dataLoader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                  shuffle=False, num_workers=0, pin_memory=True)
    dataLoader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,
                                       shuffle=False, num_workers=0, pin_memory=True)

    for idx, data in enumerate(dataLoader_train):
        for i in range(data[0].shape[0]):
            torch.save([data[0][i].to(device), data[1][i].to(device)], save_data_train_path +
                       "\\data_{}_{}".format(idx, i) + '.pth')

    for idx, data in enumerate(dataLoader_validation):
        for i in range(data[0].shape[0]):
            torch.save([data[0][i].to(device), data[1][i].to(device)], save_data_val_path +
                       "\\data_{}_{}".format(idx, i) + '.pth')


def main():
    RESNET18 = 'RES-NET-18'
    BASIC_AE = 'BASIC_AE'
    AE_RESNET18 = 'AE-RES-NET-18'
    ATTENTION_AE = 'ATTENTION_AE'
    ATTENTION_AE_RESNET18 = 'IMPROVED-AE-RES-NET-18'

    # choose configuration:
    architecture_type = RESNET18
    database_path = r'E:\AE_Classifier\dataset'
    path_img_dir = r'E:\AE_Classifier\database'
    path_file_train = database_path + '\\train_1.txt'
    path_file_validation = database_path + '\\val_1.txt'
    path_file_test = database_path + '\\test_1.txt'
    batch_size = 64

    trans_resize_size = None
    trans_crop_size = None
    trans_rotation_angle = None
    normalization_vec = None
    base_path = path_img_dir + '\\Augmented_data_' + str(architecture_type)
    save_data_train_path = base_path + '\\train'
    save_data_valid_path = base_path + '\\val'
    save_data_test_path = base_path + '\\test'
    num_of_input_channels = 1

    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if not os.path.exists(save_data_train_path):
        os.mkdir(save_data_train_path)
    if not os.path.exists(save_data_valid_path):
        os.mkdir(save_data_valid_path)
    if not os.path.exists(save_data_test_path):
        os.mkdir(save_data_test_path)

    # change input according to architecture
    if architecture_type == RESNET18:
        # resize to 256 -> random crop to 224 -> random rotate [-5,5]
        trans_resize_size = 256
        trans_crop_size = 224
        trans_rotation_angle = 5
        num_of_input_channels = 3
        normalization_vec = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif architecture_type == BASIC_AE or architecture_type == ATTENTION_AE:
        # random crop to 128
        trans_crop_size = 128
    elif architecture_type == AE_RESNET18 or architecture_type == ATTENTION_AE_RESNET18:
        # random crop to 896 -> random rotate [-5,5]
        trans_crop_size = 896
        trans_rotation_angle = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    if device == torch.device("cuda:0"):
        gc.collect()
        torch.cuda.empty_cache()
        print('Using GPU')
    else:
        print('Using CPU')

    create_dataset(path_img_dir, path_file_train, path_file_validation, batch_size,
                   trans_resize_size, trans_crop_size, trans_rotation_angle, normalization_vec,
                   save_data_train_path, save_data_valid_path, device, num_of_input_channels)


if __name__ == '__main__':
    main()
