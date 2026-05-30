
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from sklearn import svm

from utils import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from operator import truediv
import time

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

DATASET = 'Pingan'  ## Tangdaowan, Houston, Pingan, Qingyun

data, tr, te, class_name = loadData(DATASET)

#if DATASET == "Houston":
#    data = np.transpose(data, (1, 0, 2))
#    tr = tr.T
#    te = te.T
    
gt = tr+te
NUM_CLASS = gt.max()
_, labels = get_img_indexes(gt, removeZeroindexes = True)

#img_display(classes=tr,title='Training',class_name=class_name, Location = "upper left")
#img_display(classes=te,title='Testing',class_name=class_name, Location = "upper left")
#img_display(classes=gt,title='GT Full',class_name=class_name, Location = "upper left")
window_size = 1
num_PCA = 15

data = applyPCA(data, numComponents = num_PCA, normalization = True)

tr_indexes, tr_labels = get_img_indexes(tr, removeZeroindexes = True)
X_test_idx, y_test = get_img_indexes(te, removeZeroindexes = True)

# Get class map indexes
indexes, labels = get_img_indexes(gt, removeZeroindexes = True)


Aa = []
Oa = []
K = []
Ea = []
for i in range(1):
    print("Iteration number: ",i)
    X_train_idx, X_val_idx, y_train, y_val = splitTrainTestSet(tr_indexes, tr_labels, testRatio = 0.3)

    X_train = createImageCubes(data, X_train_idx, window_size)
    X_train =  np.reshape(X_train,[np.shape(X_train)[0], np.shape(X_train)[1]*np.shape(X_train)[2]*np.shape(X_train)[3]] )


    Model = svm.SVC()
    

    
    start_time = time.time()    
    Model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    
    start_time = time.time()
    Y_pred_test = predict_by_batching_SVM(Model, input_tensor_idx = X_test_idx, batch_size = 1000, X = data, windowSize = window_size)
    end_time = time.time()
    print(f"\n\nTotal Testing time: {end_time - start_time:.2f} seconds")
''' 
    kappa = cohen_kappa_score(y_test,  Y_pred_test)
    oa = accuracy_score(y_test, Y_pred_test)
    confusion = confusion_matrix(y_test, Y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    Ea.append(each_acc*100)
   
    Aa.append(float(format((aa)*100, ".2f")))
    Oa.append(float(format((oa)*100, ".2f")))
    K.append(float(format((kappa)*100, ".2f")))
    
 

print("oa = ", Oa) 
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


Predicted_Class_Map = get_class_map_svm(Model, data, gt, window_size)

Folder = 'Matlab_Outputs/'
Name = 'SVM'
sio.savemat(Folder + DATASET+'/' + Name+'.mat', {Name: Predicted_Class_Map})




'''
