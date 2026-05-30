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


DATASET = 'Pavia'  ## Pavia, Pingan, LongKou, Salinas

data, ref_data, class_name = loadData(DATASET)
num_classes = ref_data.max()
#img_display(classes=ref_data,title='GT Full',class_name=class_name, Location = "upper left")

train_num = np.min(np.bincount(ref_data.ravel())[1:])//10
val_num = train_num//5


train_im, _ = extract_connected_region_per_class(class_map = ref_data, pixels_per_class = train_num, seed=123)
test_im = ref_data - train_im
val_im, _ = extract_connected_region_per_class(class_map = test_im, pixels_per_class = val_num, seed=123)
test_im = test_im - val_im

train_indexes, train_labels = get_img_indexes(train_im, removeZeroindexes = True)
val_indexes, val_labels = get_img_indexes(val_im, removeZeroindexes = True)
test_indexes, test_labels = get_img_indexes(test_im, removeZeroindexes = True)
_, labels = get_img_indexes(ref_data, removeZeroindexes = True)

sample_report = f"{'class': ^25}{'train_num':^10}{'val_num': ^10}{'test_num': ^10}{'total': ^10}\n"
for i in range(1,num_classes+1):
    if i == 0: continue
    sample_report += f"{class_name[i]: ^25}{(train_labels==i-1).sum(): ^10}{(val_labels==i-1).sum(): ^10}{(test_labels==i-1).sum(): ^10}{(ref_data==i).sum(): ^10}\n"
sample_report += f"{'total': ^25}{len(train_labels): ^10}{len(val_labels): ^10}{len(test_labels): ^10}{len(labels): ^10}"
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
ws = 13
train_patches = generate_label_image(shape, train_indexes, train_labels, window_size=ws, background=0, order="rc", dtype=None)
test_patches = generate_label_image(shape, test_indexes, test_labels, window_size=ws, background=0, order="rc", dtype=None)

img_display(classes=train_im,title='Training Image',class_name=class_name, Location = 'lower left')
img_display(classes=train_patches,title='Training Patches',class_name=None, Location = "upper left")


img_display(classes=test_im,title=None,class_name=None, Location = "upper left")
img_display(classes=test_patches,title='Testing Patches',class_name=None, Location = "upper left")

overlap = (train_patches == test_patches)*(ref_data>0)
img_display(classes=np.uint8(overlap),title='Overlap', palette = np.array([[0, 0, 0],[255, 255, 255]]))


Leaked_Data = overlap * ref_data
img_display(classes=Leaked_Data,title='Leaked Data',class_name=None, Location = "upper left")

Overlap_per = np.sum(overlap)/np.sum(test_patches>0)
print('Overlap Percentage = ', format(Overlap_per*100, ".2f"))
