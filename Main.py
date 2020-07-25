from ModelTrainer import *


def main():

    batch_run_train(lrs = [1e-3, 1e-4, 1e-5],weight_decays = [5e-5, 1e-4, 5e-4], lambda_losses = [0, 0.3, 0.6, 0.9, 1])
    # batch_run_train(lrs = [1e-4],weight_decays = [1e-4], lambda_losses = [0, 1], max_epochs=[1])

    # run_parameters = parameters()
    # run_train(run_parameters)
    # run_test()

def batch_run_train(lrs = [1e-4],weight_decays =[1e-5],decay_patiences = [3],lambda_losses = [0.9],decay_factors = [0.1], batch_sizes=[64], max_epochs=[20]):

    if not os.path.exists(r"./ResultSummary.txt"):
        file1 = open(r"./ResultSummary.txt",'w')
        file1.write('time path weight_decay decay_patience lambda_loss decay_factor batch_size max_epoch AUROC_mean loss_train loss_val loss_test\n')
        file1.close()

    #
    # workbook = xlsxwriter.Workbook(launch_timestamp+'.xlsx')
    # worksheet = workbook.add_worksheet()
    # worksheet.write('A1', 'Path')
    # worksheet.write('B1', 'weight_decay')
    # worksheet.write('C1', 'decay_patience')
    # worksheet.write('D1', 'lambda_loss')
    # worksheet.write('E1', 'decay_factor')
    # worksheet.write('F1', 'batch_size')
    # worksheet.write('G1', 'max_epoch')
    # worksheet.write('H1', 'AUROC mean')

    runs_parameters = []
    for lr in lrs:
        for weight_decay in weight_decays:
            for decay_patience in decay_patiences:
                for lambda_loss in lambda_losses:
                    for decay_factor in decay_factors:
                        for batch_size in batch_sizes:
                            for max_epoch in max_epochs:
                                runs_parameters.append(parameters(lr=lr, weight_decay=weight_decay, decay_patience=decay_patience,
                                                        lambda_loss=lambda_loss,decay_factor=decay_factor,batch_size=batch_size, max_epoch=max_epoch))
    i=2
    for run_parameters in runs_parameters:
        print('******* Parameter check ',i-1,'/',len(runs_parameters),'*******')
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        launch_timestamp = timestampDate + '-' + timestampTime
        AUROC_mean, train_loss, val_loss, test_loss, path_model =  run_train(run_parameters)
        # worksheet.write('A'+str(i), path_model)
        # worksheet.write('B'+str(i), run_parameters.weight_decay)
        # worksheet.write('C'+str(i), run_parameters.decay_patience)
        # worksheet.write('D'+str(i), run_parameters.lambda_loss)
        # worksheet.write('E'+str(i), run_parameters.decay_factor)
        # worksheet.write('F'+str(i), run_parameters.batch_size)
        # worksheet.write('G'+str(i), run_parameters.max_epoch)
        # worksheet.write('H'+str(i), auroc_mean)
        file1 = open(r"./ResultSummary.txt",'a')
        file1.write(launch_timestamp+' '+path_model+' '+ str(run_parameters.weight_decay)+' '+
                    str(run_parameters.decay_patience)+' '+str(run_parameters.lambda_loss)+' '+str(run_parameters.decay_factor)+' '+str(run_parameters.batch_size)+' '
                    +str(run_parameters.max_epoch)+' '+str(AUROC_mean)+' '+str(train_loss)+' '+str(val_loss)+' '+str(test_loss)+'\n')
        file1.close()
        i+=1
    # workbook.close()


def run_train(run_parameters):

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

    # ---- Training settings: batch size, maximum number of epochs
    batch_size = run_parameters.batch_size
    max_epoch = run_parameters.max_epoch
    
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
    model_trainer = ModelTrainer(architecture_type, num_of_input_channels, is_backbone_pretrained, NUM_CLASSES, device, run_parameters)
    train_loss, val_loss = model_trainer.train(PATH_IMG_DIR, PATH_FILE_TRAIN, PATH_FILE_VALIDATION, batch_size,
                            max_epoch, trans_resize_size, trans_crop_size, trans_rotation_angle, launch_timestamp,
                            checkpoint_classifier, checkpoint_encoder, checkpoint_combined)
    print ('Testing the trained model')
    auroc_mean, test_loss = model_trainer.test(PATH_IMG_DIR, PATH_FILE_TEST, path_saved_model,
                       batch_size, trans_resize_size, trans_crop_size)
    return auroc_mean, train_loss, val_loss, test_loss, path_saved_model

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
    num_of_input_channels = 1
    trans_resize_size = None
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
    # folder_models = r"F:\4e5"
    path_trained_models = [path_trained_model]
    # for f in os.listdir(folder_models):
    #     name, ext = os.path.splitext(f)
    #     if ext == '.tar':
    #         path_trained_models.append(folder_models+'\\'+f)
    auroc_means = []
    for path_trained_model in path_trained_models:
        model_trainer = ModelTrainer(architecture_type, num_of_input_channels, is_backbone_pretrained, NUM_CLASSES, device)
        auroc_mean, test_loss =model_trainer.test(PATH_IMG_DIR, PATH_FILE_TEST, path_trained_model,
                            batch_size, trans_resize_size, trans_crop_size)
        auroc_means.append(auroc_mean)
    best_model = np.argmax(auroc_means)
    modelCheckpoint = torch.load(path_trained_models[best_model])
    decay = modelCheckpoint['optimizer']['param_groups'][0]['weight_decay']
    lr = modelCheckpoint['optimizer']['param_groups'][0]['lr']
    torch.save(modelCheckpoint,'m-' + modelCheckpoint['model_type'] + '-' + str(decay) + '-' + str(lr) + '-' + str(np.round(10000*auroc_means[best_model])/10000) + '.pth.tar')
if __name__ == '__main__':
    main()






