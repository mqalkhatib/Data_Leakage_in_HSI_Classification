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
from sklearn import svm

DATASET = 'Pingan '  ## Pavia, Pingan, LongKou, Salinas

data, ref_data, class_name = loadData(DATASET)
num_classes = ref_data.max()
img_display(classes=ref_data,title='GT Full',class_name=class_name, Location = "upper left")

train_num = np.min(np.bincount(ref_data.ravel())[1:])//10
val_num = train_num//5

idx, labels = get_img_indexes(ref_data, removeZeroindexes = True)

train_idx, yTrain, test_idx, yTest = getTrainTestSplit(idx, labels, train_num, SeeD = randint(0, 1000))
val_idx, yVal, test_idx, yTest = getTrainTestSplit(test_idx, np.array(yTest), val_num, SeeD = randint(0, 1000))

yTrain = np.array(yTrain)
yTest = np.array(yTest)
yVal = np.array(yVal)
sample_report = f"{'class': ^25}{'train_num':^10}{'val_num': ^10}{'test_num': ^10}{'total': ^10}\n"
for i in range(1,num_classes+1):
    if i == 0: continue
    sample_report += f"{class_name[i]: ^25}{(yTrain==i-1).sum(): ^10}{(yVal==i-1).sum(): ^10}{(yTest==i-1).sum(): ^10}{(ref_data==i).sum(): ^10}\n"
sample_report += f"{'total': ^25}{len(yTrain): ^10}{len(yVal): ^10}{len(yTest): ^10}{len(labels): ^10}"
print(sample_report)



train_im = build_image_from_indices(train_idx, yTrain, ref_data.shape)
val_im = build_image_from_indices(val_idx, yVal, ref_data.shape)
test_im = build_image_from_indices(test_idx, yTest, ref_data.shape)


img_display(classes=train_im,title='Training Image',class_name=class_name, Location = "upper left")
img_display(classes=val_im,title='Validation Image',class_name=class_name, Location = "upper left")
img_display(classes=test_im,title='Testing Image',class_name=class_name, Location = "upper left")
#################################################
window_size = 1
num_PCA = 15

data = applyPCA(data, numComponents = num_PCA, normalization = True)

x_train = createImageCubes(data, train_idx, windowSize = window_size)
x_val = createImageCubes(data, val_idx, windowSize = window_size)


Aa = []
Oa = []
K = []
Ea = []
seeds = [10, 15, 45, 123, 345, 321, 543, 1, 938, 666]
for i in range(1):
    print("Iteration number: ",i)
    ##################################################################################
    # Get another distribution
    train_idx, yTrain, test_idx, yTest = getTrainTestSplit(idx, labels, train_num, SeeD = seeds[i])
    val_idx, yVal, test_idx, yTest = getTrainTestSplit(test_idx, np.array(yTest), val_num, SeeD = seeds[i])
    
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)
    yVal = np.array(yVal)
    
    X_train = createImageCubes(data, train_idx, windowSize = window_size)
    x_train = np.reshape(X_train,[np.shape(X_train)[0], np.shape(X_train)[1]*np.shape(X_train)[2]*np.shape(X_train)[3]] )

    y_train = keras.utils.to_categorical(yTrain)
    x_val = createImageCubes(data, val_idx, windowSize = window_size)
    y_val = keras.utils.to_categorical(yVal)
    ###################################################################################    
    
    
    model = svm.SVC()
    
    model.fit(x_train, yTrain)
    
    Y_pred = predict_by_batching_SVM(model, input_tensor_idx = test_idx, batch_size = 1000, X = data, windowSize = window_size)
    #Y_pred = model.predict(x_test)
    y_pred = Y_pred
    confusion = confusion_matrix(yTest, y_pred)
    oa = accuracy_score(yTest, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(yTest, y_pred)
    print("\noa = ", oa) 
    print("aa = ", aa)
    print('Kappa = ', kappa)

   
    Aa.append(float(format((aa)*100, ".2f")))
    Oa.append(float(format((oa)*100, ".2f")))
    K.append(float(format((kappa)*100, ".2f")))
    Ea.append(each_acc*100)
 

print("\n\noa = ", Oa) 
print("aa = ", Aa)
print('Kappa = ', K)
print('\n')
print('Mean OA = ', format(np.mean(Oa), ".2f"), '+', format(np.std(Oa), ".2f"))
print('Mean AA = ', format(np.mean(Aa), ".2f"), '+', format(np.std(Aa), ".2f"))
print('Mean Kappa = ', format(np.mean(K), ".2f"), '+', format(np.std(K), ".2f"))
EA_mean = np.mean(Ea, axis  = 0)
EA_mean = [round(item, 2) for item in EA_mean]
EA_std = np.std(Ea, axis = 0)
EA_std = [round(item, 2) for item in EA_std]




def get_class_map_svm(model, X, label, window_size):
    indexes, labels = get_img_indexes(label, removeZeroindexes = False)
    y_pred = predict_by_batching_SVM(model, indexes, 10000, X, window_size)
    Y_pred = np.reshape(y_pred, label.shape) + 1
    
    return Y_pred

Predicted_Class_Map = get_class_map_svm(model, data, ref_data, window_size)
gt_binary = ref_data.copy()
gt_binary[gt_binary>0]=1
img_display(classes=Predicted_Class_Map*gt_binary,title='Predicted Class Map',class_name=class_name, Location = "upper left")




Folder = 'Matlab_Outputs/'
Name = 'SVM'
sio.savemat(Folder + DATASET+'/Random_Sampling/' + Name+'.mat', {Name: Predicted_Class_Map})























