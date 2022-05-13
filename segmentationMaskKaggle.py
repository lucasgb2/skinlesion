#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 10:15:03 2021

@author: lucas
"""

import os
import numpy as np
import pprint
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import is_low_contrast
from sklearn.metrics import  jaccard_score
from scipy import spatial
from skimage import io, transform, color, filters, img_as_ubyte
import constHam1000 as ham
import pprint as pp
import imgaug as ia
import imgaug.augmenters as iaa
import timeit
from tqdm import tqdm
import threading
import concurrent.futures

      
def redimensionando():
    path = os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, 'train', 'image')
    filesTrain = os.listdir(path) 
    
    for img in filesTrain:
        ham.resizeImage(os.path.join(path, img))
        
    path = os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, 'train', 'label')
    filesTrain = os.listdir(path) 
    
    for img in filesTrain:
        ham.resizeImage(os.path.join(path, img))        

# Aqui é formalizando a binarização das imagens anotadas que serão utilizadas para treino. 
# O framework exige que as imagens possuam pixel 0 ou 1. Alguns pixels da image não são 1. 
# Aqui é feito este ajuste forçado nas imagens de treino setando 1 para pixels diferente de 0"""
def transformBinaryImageTrain():
  filesTrain = os.listdir(os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, 'train', 'label')) 
  
  def transform(n):
      nameImg =  os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, 'train', 'label', n)
      img = np.array(io.imread(nameImg))
      for i in range(img.shape[0]): #para cada coluna
          for j in range(img.shape[1]): #para cada linha
              if img[i,j] > 0:
                  img[i,j] = 1
      io.imsave(os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, 'train', 'label', n), img, check_contrast=False)
      print('Pronto:'+n)

  #Colocando os pixels que são maior que 0 como 1. Framework exige isto  
  for img in tqdm(filesTrain):
      #pool.map(transform(img))      
      transform(img)
  


def plotStepByStep(original, mask, segmented, name):
  figure, axis = plt.subplots(1,2, figsize=(224,224))
  axis[0].imshow(original)
  axis[1].imshow(segmented)  
  axis[0].axis('off')
  axis[1].axis('off')

 # figure.savefig(os.path.join(ham.IMAGES_SEGMENTED, 'plot', name))

def segmentarImagens():    
    filesOriginal = []
    filesOriginal = os.listdir(ham.IMAGES_ORIGINAL)  
        
    for f in filesOriginal:      
      img = os.path.join(ham.IMAGES_ORIGINAL, f) 
      
      if not os.path.exists(os.path.join(ham.HAM10000, 'HAM10000_segmentations_lesion_tschandl', f.replace('jpg','png'))):
          print('Mask not exists for image: '+f)
          continue
      
      mask = os.path.join(ham.HAM10000, 'HAM10000_segmentations_lesion_tschandl', f.replace('jpg','png'))
    
      mask = io.imread(mask)
      mask = np.asarray(mask)
      mask = color.rgb2gray(mask)
      
      img = io.imread(img)
      img = np.asarray(img) 
    
      try:
        t = filters.threshold_otsu(mask)
        mask[mask <= t] = 0
        mask[mask > t] = 1
        mask = color.gray2rgb(mask)
        maskInt = mask.astype(int)  
    
        imgSegmented = maskInt * img    
        imgSegmented = imgSegmented.astype(np.uint8)        
        imgSegmented = transform.resize(imgSegmented, (224,224)) #redimensionando a imagem            
        imgSegmented = img_as_ubyte(imgSegmented)
        #plotStepByStep(img, mask, imgSegmented, f)   
        io.imsave(os.path.join(ham.HAM10000,'segmentation2', 'without-agu', f), imgSegmented, check_contrast=False)                
      except Exception as e:
        print('Não segmentada '+f)
        print(e)
      
        
        


#transformBinaryImageTrain()
#redimensionando()

print('SEGMENTANDO')
segmentarImagens()