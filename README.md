# AE_Classifier_repo
Attentive input scaling for Chest X-ray14 classification with class-balanced loss

by Paz Ilan and Omri Levi (TAU, Deep Learning in Medical Imaging by Hayit Greenspan, 0553-5542, Spring 2020) 

## 1. Introduction:
This repository includes the code for training and testing the the models described in the report.
It consist of main models:
- ResNet-18 - Basic classifier
- BasicAutoEncoder - basic auto-encoder as stated in [1]
- ImprovedAutoEncoder - improvement of the auto-encoder as stated in our report.
- AE_Resnet18 - the basic auto-encoder combined with Resnet18 - as stated in [1].
- IMPROVED_AE_Resnet18 - the improved auto-encoder combined with Resnet18 as stated in our report.
- AttentionUnet2D - attention U-Net as stated in [2].
- AttentionUnetResnet18 - attention U-Net combined with Resnet18 as stated in our report.
This repository is for personal educational use only.

## 2. Dataset:
Download Chest X-ray14 dataset from here:

https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

## 3. Requirements:
Use the yml file (AE_Classifier/environment_requirements.yml) to see the requirements (using anaconda it is easier to set the environment)

## 4. How to run:
Every model can be trained or tested. The training process can begin from scratch or from pre-trained parameters.
- In Config.py - set the pathes for dataset images path and train, validation and test files.
- In Main.py - one should select between: batch_run_train, run_train and run_test:

    - batch_run_train - will run training with different configurations of hyper parameters
    - run_train - will run training with a specific configuration of hyper parameter and testing at the end of the training. In this function you should set the following:
	    - architecture_type - one of the following: RESNET18, BASIC_AE, AE_RESNET18, IMPROVED_AE, IMPROVED_AE_RESNET18, ATTENTION_AE, ATTENTION_AE_RESNET18
	    - is_backbone_pretrained - for Resnet18 training
	    - balanced_classifier_loss - if "True" - will set the classifier loss to be balanced-BCE loss.
	    - checkpoint_encoder, checkpoint_classifier, checkpoint_combined - check point pathes for each module (continue training from checkpoint).
    - run_test - will run testing. Should set the following:
	    - architecture_type, is_backbone_pretrained, balanced_classifier_loss - as in "run_train"
	    - path_trained_model - path to the trained model.
	
## 5. Credits and References:
- [1] Ranjan, Ekagra, et al. "Jointly Learning Convolutional Representations to Compress Radiological Images and Classify Thoracic Diseases in the Compressed Domain." Proceedings of the 11th Indian Conference on Computer Vision, Graphics and Image Processing. 2018.‏ 
		https://github.com/ekagra-ranjan/AE-CNN
- [2] Oktay, Ozan, et al. "Attention u-net: Learning where to look for the pancreas." arXiv preprint arXiv:1804.03999 (2018).‏ 
		https://github.com/ozan-oktay/Attention-Gated-Networks.git
- [3] https://github.com/zoogzog/chexnet.git