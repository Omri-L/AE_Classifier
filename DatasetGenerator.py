import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform, num_img_chs=1, num_labels=14):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        self.num_img_chs = num_img_chs
        self.num_sample_per_label = [0] * num_labels


        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                for label_idx, label in enumerate(imageLabel):
                    self.num_sample_per_label[label_idx] += label
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]

        if self.num_img_chs == 3:
            imageData = Image.open(imagePath).convert('RGB')
        else:
            imageData = Image.open(imagePath).convert('L')

        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)

    def get_num_samples_in_label(self, label_idx):
        return self.num_sample_per_label[label_idx]
    
 #-------------------------------------------------------------------------------- 
    