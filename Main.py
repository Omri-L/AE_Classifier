from ModelTrainer import *


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
    architecture_type = AE_RESNET18
    is_backbone_pretrained = True
    num_classes = 14
    
    # ---- Training settings: batch size, maximum number of epochs
    batch_size = 64
    max_epoch = 30
    
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
    checkpoint_encoder = r"./m-BASIC_AE.pth.tar"
    checkpoint_classifier = r"./m-RES-NET-18.pth.tar"
    checkpoint_combined = None

    print('Training NN architecture = ', architecture_type)
    model_trainer = ModelTrainer(architecture_type, num_of_input_channels, is_backbone_pretrained, num_classes, device)
    model_trainer.train(PATH_IMG_DIR, PATH_FILE_TRAIN, PATH_FILE_VALIDATION, batch_size,
                        max_epoch, trans_resize_size, trans_crop_size, trans_rotation_angle, launch_timestamp,
                        checkpoint_classifier, checkpoint_encoder, checkpoint_combined)
    print ('Testing the trained model')
    model_trainer.test(PATH_IMG_DIR, PATH_FILE_TEST, path_saved_model,
                       batch_size, trans_resize_size, trans_crop_size)


def run_test():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device == torch.device("cuda:0"):
        gc.collect()
        torch.cuda.empty_cache()
        print('Using GPU')
    else:
        print('Using CPU')

    architecture_type = AE_RESNET18  # select from: RESNET18, AE_RESNET18, IMPROVED_AE_RESNET18
    is_backbone_pretrained = True
    num_classes = 14
    if architecture_type == RESNET18:
        # resize to 256 -> random crop to 224 -> random rotate [-5,5]
        batch_size = 1024
        trans_resize_size = 256
        trans_crop_size = 224
        num_of_input_channels = 3
    elif architecture_type == BASIC_AE or architecture_type == ATTENTION_AE:
        batch_size = 1024
        # random crop to 128
        trans_crop_size = 128
    elif architecture_type == AE_RESNET18 or architecture_type == ATTENTION_AE_RESNET18:
        # random crop to 896 -> random rotate [-5,5]
        batch_size = 256
        trans_crop_size = 896



    path_trained_model = r'C:\Users\pazi\Desktop\Uni\BioDeepLearning\1e4\m-RES-NET-18-20072020-073848.pth.tar'
    folder_models = r"F:\4e5"
    path_trained_models = []
    for f in os.listdir(folder_models):
        name, ext = os.path.splitext(f)
        if ext == '.tar':
            path_trained_models.append(folder_models+'\\'+f)
    auroc_means = []
    for path_trained_model in path_trained_models:
        model_trainer = ModelTrainer(architecture_type, num_of_input_channels, is_backbone_pretrained, num_classes, device)
        auroc_means.append(model_trainer.test(PATH_IMG_DIR, PATH_FILE_TEST, path_trained_model,
                            batch_size, trans_resize_size, trans_crop_size))
    best_model = np.argmax(auroc_means)
    modelCheckpoint = torch.load(path_trained_models[best_model])
    decay = modelCheckpoint['optimizer']['param_groups'][0]['weight_decay']
    lr = modelCheckpoint['optimizer']['param_groups'][0]['lr']
    torch.save(modelCheckpoint,'m-' + modelCheckpoint['model_type'] + '-' + str(decay) + '-' + str(lr) + '.pth.tar' + '-' + str(np.round(1000*auroc_means[best_model])/1000))
if __name__ == '__main__':
    main()






