import os
import numpy as np
import pprint
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import is_low_contrast
from keras_segmentation.models.unet import vgg_unet, unet
from keras_segmentation.predict import predict, predict_multiple
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


def renomear():
  filesTrain = os.listdir(os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, 'train', 'label'))  
  for n in filesTrain:
      nameImg =  os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, 'train', 'label', n)
      nn, extension = os.path.splitext(nameImg)
      nn = n[0:12] + extension
      os.rename(nameImg, os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, 'train', 'label', nn))
      
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
  



def getModelTrained():
  model = unet(n_classes=2 ,  input_height=224, input_width=224 )
  model.train(
    train_images = os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, 'train', 'image'),
    train_annotations = os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, 'train', 'label'),
    checkpoints_path = os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, "unet_1")  , 
    epochs=30
  )
  
  
  k = model.evaluate_segmentation(inp_images_dir = os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, 'test','image'),  
                                  annotations_dir=os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, 'test','label'))  
  print('AVALIAÇÃO DO MODELO')
  print('---------------------------')
  pp.pprint(k)
  print('---------------------------')

  return model

def plotStepByStep(original, mask, segmented, name):
  figure, axis = plt.subplots(1,3, figsize=(15,15))
  axis[0].imshow(original)
  axis[1].imshow(mask)
  axis[2].imshow(segmented)
  figure.savefig(os.path.join(ham.IMAGES_SEGMENTED, 'plot', name))

def segmentarImagens():
    filesMask = []
    filesMask = os.listdir(os.path.join(ham.IMAGES_SEGMENTATION, 'without-agu', 'mask'))  
    #filesMask = filesMask[0:3]
    
    for f in tqdm(filesMask):
      mask = os.path.join(ham.IMAGES_SEGMENTATION, 'without-agu', 'mask', f)
      img = os.path.join(ham.IMAGES_ORIGINAL, f) 
    
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
        if is_low_contrast(imgSegmented):
            print('Baixa contraste: '+f)
            io.imsave(os.path.join(ham.IMAGES_SEGMENTATION, 'without-agu', 'segmented', f), img, check_contrast=False)
        else:               
            io.imsave(os.path.join(ham.IMAGES_SEGMENTATION, 'without-agu', 'segmented', f), imgSegmented, check_contrast=False)
        
      except Exception as e:
        print('Não segmentada '+f)
        print(e)


#renomear()
#transformBinaryImageTrain()
#redimensionando()

modelSeg = getModelTrained()
checkpoints_path = os.path.join(ham.IMAGES_TRAIN_SEGMENTATION, "unet_1"),
print('PREDICT')
modelSeg.predict_multiple(inp_dir = ham.IMAGES_ORIGINAL,
                         out_dir = os.path.join(ham.IMAGES_SEGMENTATION, 'without-agu', 'mask'))
print('SEGMENTANDO')
segmentarImagens()