import os
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt 
import matplotlib.patches as mpts
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from scipy.io import loadmat
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import scipy.io as sio
from keras import layers
import tensorflow_addons as tfa
from random import randint


DATASET = 'Pavia'  ## Pavia, Pingan
data, ref_data, class_name = loadData(DATASET)
num_classes = ref_data.max()
#img_display(classes=ref_data,title='GT Full',class_name=class_name, Location = "upper left")

train_num = np.min(np.bincount(ref_data.ravel())[1:])//10
val_num = train_num//5

idx, labels = get_img_indexes(ref_data, removeZeroindexes = True)

seeds = [10, 15, 45, 123, 345, 321, 543, 1, 938, 666]
leakage = []
over_lap_ratio = []
for i in range(10):        
    train_idx, yTrain, test_idx, yTest = getTrainTestSplit(idx, labels, train_num, SeeD = seeds[i])
    val_idx, yVal, test_idx, yTest = getTrainTestSplit(test_idx, np.array(yTest), val_num, SeeD = seeds[i])
    
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)
    yVal = np.array(yVal)
    
    train_im = build_image_from_indices(train_idx, yTrain, ref_data.shape)
    val_im = build_image_from_indices(val_idx, yVal, ref_data.shape)
    test_im = build_image_from_indices(test_idx, yTest, ref_data.shape)
    
    #img_display(classes=train_im,title='Training Image',class_name=None, Location = "upper left")
    
    #img_display(classes=test_im,title='Testing Image',class_name=None, Location = "upper left")
    
    
    tmp, tmp2 = calculate_same_class_leakage(train_image = train_im, 
                                                      test_image = test_im, 
                                                      patch_size = 5)
    leakage.append(tmp)
    over_lap_ratio.append(tmp2)
    
print('Mean Leakage = ', format(np.mean(leakage), ".2f"), '\u00B1', format(np.std(leakage), ".2f"))    
print('Average Overlap Ratio = ', format(np.mean(over_lap_ratio), ".2f"), '\u00B1', format(np.std(over_lap_ratio), ".2f"))    
    
#print(f"Same-class leakage: {leakage:.2f}%")
