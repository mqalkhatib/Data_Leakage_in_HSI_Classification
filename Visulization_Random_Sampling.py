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

train_idx, yTrain, test_idx, yTest = getTrainTestSplit(idx, labels, train_num, SeeD = 123)
val_idx, yVal, test_idx, yTest = getTrainTestSplit(test_idx, np.array(yTest), val_num, SeeD =123)

yTrain = np.array(yTrain)
yTest = np.array(yTest)
yVal = np.array(yVal)

train_im = build_image_from_indices(train_idx, yTrain, ref_data.shape)
val_im = build_image_from_indices(val_idx, yVal, ref_data.shape)
test_im = build_image_from_indices(test_idx, yTest, ref_data.shape)

sample_report = f"{'class': ^25}{'train_num':^10}{'val_num': ^10}{'test_num': ^10}{'total': ^10}\n"
for i in range(1,num_classes+1):
    if i == 0: continue
    sample_report += f"{class_name[i]: ^25}{(yTrain==i-1).sum(): ^10}{(yVal==i-1).sum(): ^10}{(yTest==i-1).sum(): ^10}{(ref_data==i).sum(): ^10}\n"
sample_report += f"{'total': ^25}{len(yTrain): ^10}{len(yVal): ^10}{len(yTest): ^10}{len(labels): ^10}"
print(sample_report)

########################################################################################################
def generate_label_image(shape, indices, labels, window_size=15, background=0, order="rc", dtype=None):
    img = np.full(shape, background, dtype=dtype if dtype is not None else None)

    half_w = window_size // 2

    for (idx, label) in zip(indices, labels):
        r, c = idx

        # Define window boundaries (clamped to image size)
        r_start, r_end = max(0, r - half_w), min(shape[0], r + half_w + 1)
        c_start, c_end = max(0, c - half_w), min(shape[1], c + half_w + 1)

        img[r_start:r_end, c_start:c_end] = label + 1

    return img

#########################################################################################################
shape   = data.shape[:2]
ws = 11
train_patches = generate_label_image(shape, train_idx, yTrain, window_size=ws, background=0, order="rc", dtype=None)
test_patches = generate_label_image(shape, test_idx, yTest, window_size=ws, background=0, order="rc", dtype=None)

img_display(classes=train_im,title='Training Image',class_name=None, Location = "upper left")
img_display(classes=train_patches,title='Training Patches',class_name=None, Location = "upper left")


img_display(classes=test_im,title='Testing Image',class_name=None, Location = "upper left")
img_display(classes=test_patches,title='Testing Patches',class_name=None, Location = "upper left")

overlap = (train_patches == test_patches)*(ref_data>0)
img_display(classes=np.uint8(overlap),title='Overlap', palette = np.array([[0, 0, 0],[255, 255, 255]]))


Leaked_Data = overlap * ref_data
img_display(classes=Leaked_Data,title='Leaked Data',class_name=None, Location = "upper left")

Overlap_per = np.sum(overlap)/np.sum(test_patches>0)
print('Overlap Percentage = ', format(Overlap_per*100, ".2f"))
