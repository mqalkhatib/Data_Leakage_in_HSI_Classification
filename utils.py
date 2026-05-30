import os
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
import spectral
import matplotlib.pyplot as plt
from operator import truediv
###########################################################################################
def loadData(name): ## customize data and return data label and class_name
    data_path = os.path.join(os.getcwd(),'datasets')

    if name == 'Pingan':
        data = loadmat(os.path.join(data_path, 'QUH-Pingan.mat'))['Haigang']
        gt = loadmat(os.path.join(data_path, 'QUH-Pingan_GT.mat'))['HaigangGT']
        class_name = ["Unassigned", "Ship", "Seawater", "Trees"," Concrete structure bldng", "Floating pier", "Brick houses",
                      "Steel houses"," Wharf construction land", "Car", "Road"]
    
    elif name == 'Pavia':
        data = loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        gt = loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        class_name = ['Unassigned', 'Asphalt', 'Meadows', 'Gravel', 'Trees', 'Metal Sheet', 
                      'Bare Soil', 'Bitumen', 'Brick', 'Shadow']
    
    elif name == 'LongKou':
        data = loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        gt = loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
        class_name = ['Unassigned', 'Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean', 
                  'Narrow-leaf soybean', 'Rice', 'Water', 'Roads and houses', 'Mixed weed']
        
    elif name == 'Salinas':
         data = loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
         gt = loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
         class_name = ['Unassigned', 'Brocoli-green-weeds-1 ', 'Brocoli-green-weeds-2', 'Fallow ', 
                       'Fallow-rough-plow', 'Fallow-smooth', 'Stubble', 'Celery', 'Grapes-untrained ', 
                       'Soil-vinyard-develop', 'Corn-senesced-green-weeds', 'Lettuce-romaine-4wk', 
                       'Lettuce-romaine-5wk', 'Lettuce-romaine-6wk', 'Lettuce-romaine-7wk', 
                       'Vinyard-untrained  ','Vinyard-vertical-trellis ']
    return data, gt, class_name

###########################################################################################
def get_img_indexes (class_map, removeZeroindexes = True):
    """
    Get indices of elements in the class map.
    
    Parameters:
    class_map (numpy array): The class map (2D array).
    removeZero (bool): If True, return indices of non-zero elements, 
                       otherwise return indices of all elements.
    
    Returns:
    tuple: (indices, labels)
           - indices: List of tuples representing the indices of the selected elements.
           - labels: Array of labels corresponding to the indices.
    """
    if removeZeroindexes:
        # Get indices of non-zero values
        indices = np.argwhere(class_map != 0)
    else:
        # Get indices of all elements (including zeros)
        indices = np.argwhere(class_map != None)
    
    # Flatten the class map to get the corresponding pixel values (labels)
    labels = class_map[indices[:, 0], indices[:, 1]]
    
    # Convert indices to a list of tuples for easier use
    indices = [tuple(idx) for idx in indices]
    
    return indices, np.array(labels.tolist()) - 1

def createImageCubes(X, indices, windowSize):
    """
    Extract patches centered at given indices from the hyperspectral image 
    after applying zero padding.
    
    Parameters:
    X (numpy array): Hyperspectral image of shape (N, M, P)
    indices (list of tuples): List of indices where patches should be extracted
    windowSize (int): Window size, the patch will be of size (windowSize, windowSize)
    
    Returns:
    list: List of image patches extracted from the padded hyperspectral image
    """
    # Calculate margin based on window size
    margin = windowSize // 2
    
    # Apply zero padding to the hyperspectral image
    N, M, P = X.shape
    X_padded = np.zeros((N + 2 * margin, M + 2 * margin, P))
    
    # Offsets to place the original image in the center of the padded image
    x_offset = margin
    y_offset = margin
    X_padded[x_offset:N + x_offset, y_offset:M + y_offset, :] = X
    
    # Extract patches centered at the provided indices
    patches = []
    
    for idx in indices:
        i, j = idx
        i = i + margin
        j = j + margin
        # Get patch boundaries, ensuring the patch is centered at (i, j)
        i_min = i - margin  # Centered on the index, accounting for padding
        i_max = i_min + windowSize
        j_min = j - margin
        j_max = j_min + windowSize
        
        # Extract the patch
        patch = X_padded[i_min:i_max, j_min:j_max, :]
        

        patches.append(patch)
    
    return np.array(patches)

###########################################################################################
def splitTrainTestSet(X, y, testRatio):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, 
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

###########################################################################################    
def applyPCA(X, numComponents=15, normalization = True):
    """PCA and processing
    Args:
        X (ndarray M*N*C): data needs DR
        numComponents (int, optional):  number of reserved components(Defaults to 15, 0 for no PCA).
        norm: normalization or not
    Returns:
        newX: processed data
        pca:
    """

    if numComponents == 0:
        newX = np.reshape(X, (-1, X.shape[2]))
    else:
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents)   ##PCA and normalization
        newX = pca.fit_transform(newX)
    if normalization:
        newX = minmax_scale(newX, axis=1)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], -1))
    return newX

###########################################################################################


def predict_by_batching(model, input_tensor_idx, batch_size, X, windowSize):
    '''
    Function to to perform predictions by dividing large tensor into small ones 
    to reduce load on GPU
    
    Parameters
    ----------
    model: The model itself with pre-trained weights.
    input_tensor: Tensor of diemnsion batches x windowSize x windowSize x channels x 1.
    batch_size: integer value smaller than batches .

    Returns
    -------
    Predicetd labels
    '''
    
    num_samples = len(input_tensor_idx)
    k = 0
    predictions = []
    for i in tqdm(range(0, num_samples, batch_size), desc="Progress"):
        k+=1
        
        batch = createImageCubes(X, input_tensor_idx[i:i + batch_size], windowSize)
        batch_predictions = model.predict(batch, verbose=0)
        predictions.append(batch_predictions)
        
    Y_pred_test = np.concatenate(predictions, axis=0)
  
    return Y_pred_test

###########################################################################################
def get_class_map(model, X, label, window_size):
    indexes, labels = get_img_indexes(label, removeZeroindexes = False)
    
    y_pred = predict_by_batching(model, indexes, 10000, X, window_size)
    
    y_pred = (np.argmax(y_pred, axis=1)).astype(np.uint8)
    
    Y_pred = np.reshape(y_pred, label.shape) + 1
    
   
    return Y_pred

###########################################################################################
import matplotlib.patches as mpatches
def img_display(data = None, rgb_band = None, classes = None,class_name = None,title = None, 
                figsize = (7,7),palette = spectral.spy_colors, Location = 'upper left'):
    if data is not None:
        im_rgb = np.zeros_like(data[:,:,0:3])
        im_rgb = data[:,:,rgb_band]
        im_rgb = im_rgb/(np.max(np.max(im_rgb,axis = 1),axis = 0))*255
        im_rgb = np.asarray(im_rgb,np.uint8)
        fig, rgbax = plt.subplots(figsize = figsize)
        rgbax.imshow(im_rgb)
        rgbax.set_title(title)
        rgbax.axis('off')
        
    elif classes is not None:
        rgb_class = np.zeros((classes.shape[0],classes.shape[1],3))
        for i in np.unique(classes):
            rgb_class[classes==i]=palette[i]
        rgb_class = np.asarray(rgb_class, np.uint8)
        _,classax = plt.subplots(figsize = figsize)
        classax.imshow(rgb_class)
        classax.set_title(title)
        classax.axis('off')
        
    if class_name is not None:
        # Create legend patches
        patches = []
        for i in np.unique(classes):
            color = np.array(palette[i]) / 255.0  # normalize RGB
            patches.append(mpatches.Patch(color=color, label=class_name[i]))

        # Add legend next to image
        #classax.legend(handles=patches, bbox_to_anchor=(1.05, 1),loc= Location, borderaxespad=0., 
        #               fontsize=15, title="Classes")
        classax.legend(
            handles=patches,
            bbox_to_anchor=(0.5, -0.05),
            loc='upper center',
            borderaxespad=0.,
            fontsize=15,
            title="Classes",
            ncol=4
                )

    plt.tight_layout()
        


def display_history(history):
    # Retrieve loss and accuracy data
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    
    # Create a figure with 2 horizontal subplots
    plt.figure(figsize=(12, 5))
    
    # Subplot for training and validation loss
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot for training and validation accuracy
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    # Show the combined figure
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    
    
    # Get training history
    val_acc = history.history['val_accuracy']
    best_epoch = np.argmax(val_acc)
    best_val = val_acc[best_epoch]
    
    # Plot Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], 'y', label='Train Acc')
    plt.plot(val_acc, 'r', label='Val Acc')
    
    # Mark best epoch
    plt.axvline(best_epoch, color='k', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.scatter(best_epoch, best_val, color='black')
    plt.text(best_epoch, best_val, f"{best_val:.2f}", fontsize=10, color='black', va='bottom')
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



from collections import deque
def extract_connected_region_per_class(class_map, pixels_per_class, seed=None, ignore_labels=(0,), connectivity=8):
    """
    Extract one connected region per class with a fixed number of pixels.

    Parameters
    ----------
    class_map : np.ndarray
        2D array of shape (H, W) containing class labels.
    pixels_per_class : int
        Number of pixels to extract for each class.
    seed : int or None
        Random seed for reproducibility.
    ignore_labels : tuple
        Labels to ignore, such as background.
    connectivity : int
        Either 4 or 8 for neighborhood connectivity.

    Returns
    -------
    region_map : np.ndarray
        Same shape as class_map. Selected pixels keep their class labels,
        all other pixels are set to 0.
    info : dict
        Metadata for each class.
    """

    rng = np.random.default_rng(seed)
    H, W = class_map.shape
    region_map = np.zeros_like(class_map)

    classes = np.unique(class_map)
    classes = [c for c in classes if c not in ignore_labels]

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        raise ValueError("connectivity must be either 4 or 8")

    info = {}

    for cls in classes:
        class_pixels = np.argwhere(class_map == cls)

        if len(class_pixels) == 0:
            continue

        # Shuffle candidate seeds until one grows enough pixels
        candidate_indices = rng.permutation(len(class_pixels))
        selected_region = None
        selected_seed = None

        for idx in candidate_indices:
            seed_r, seed_c = class_pixels[idx]

            visited = np.zeros((H, W), dtype=bool)
            region_coords = []
            q = deque()
            q.append((seed_r, seed_c))
            visited[seed_r, seed_c] = True

            while q and len(region_coords) < pixels_per_class:
                r, c = q.popleft()

                if class_map[r, c] == cls:
                    region_coords.append((r, c))

                    # Randomize neighbor order for less directional growth
                    local_neighbors = neighbors.copy()
                    rng.shuffle(local_neighbors)

                    for dr, dc in local_neighbors:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                            visited[nr, nc] = True
                            if class_map[nr, nc] == cls:
                                q.append((nr, nc))

            if len(region_coords) >= pixels_per_class:
                selected_region = region_coords[:pixels_per_class]
                selected_seed = (int(seed_r), int(seed_c))
                break

        # If no connected component is large enough, use the largest found region
        if selected_region is None:
            # fallback: pick random pixels from the class
            chosen = class_pixels[rng.choice(len(class_pixels),
                                             size=min(pixels_per_class, len(class_pixels)),
                                             replace=False)]
            selected_region = [(int(r), int(c)) for r, c in chosen]
            selected_seed = None

        for r, c in selected_region:
            region_map[r, c] = cls

        info[cls] = {
            "seed": selected_seed,
            "num_pixels": len(selected_region)
        }

    return region_map, info




import random
from sklearn.utils import shuffle
def getTrainTestSplit(X, y, pxls_num, SeeD = 123):
    X = np.array(X)
    if type(pxls_num) != list:
        pxls_num = [pxls_num]*len(np.unique(y))
        
    if len(np.unique(y)) != len(pxls_num):
        print("length of pixels list doen't match the number of classes in the dataset")
        return
    else:
        xTrain = []
        yTrain = []
    
        xTest  = []
        yTest  = []
        for i in range(len(np.unique(y))):
            if pxls_num[i] > len(y[y==i]):
                print("Number of training pixles is larger than class pixels")
            #    return
            else:
                random.seed(SeeD) #optional to reproduce the data
                samples = random.sample(range(len(y[y==i])), pxls_num[i])
                xTrain.extend(X[y==i][samples])
                
                yTrain.extend(y[y==i][samples])
        
             
                tmp1 = list(X[y==i])
               
                tmp3 = list(y[y==i])
                for ele in sorted(samples, reverse = True):
                    del tmp1[ele]
                    del tmp3[ele]

                xTest.extend(tmp1)
                yTest.extend(tmp3)

                
    xTrain, yTrain = shuffle(xTrain, yTrain, random_state=321)  
    xTest,  yTest = shuffle(xTest, yTest, random_state=345)
       
    return xTrain, yTrain, xTest, yTest
        

def build_image_from_indices(indices, labels, image_shape):
    """
    indices: (N, 2) array → [row, col]
    labels:  (N,) array
    image_shape: (H, W)
    """

    image = np.zeros(image_shape, dtype=np.uint8)
    
    indices = np.array(indices)
    labels = np.array(labels) + 1
   
    rows = indices[:, 0]
    cols = indices[:, 1]

    image[rows, cols] = labels

    return image


def predict_by_batching_SVM(model, input_tensor_idx, batch_size, X, windowSize):
    '''
    Function to to perform predictions by dividing large tensor into small ones 
    to reduce load on GPU
    
    Parameters
    ----------
    model: The model itself with pre-trained weights.
    input_tensor: Tensor of diemnsion batches x windowSize x windowSize x channels x 1.
    batch_size: integer value smaller than batches .

    Returns
    -------
    Predicetd labels
    '''
    
    num_samples = len(input_tensor_idx)
    k = 0
    predictions = []
    for i in tqdm(range(0, num_samples, batch_size), desc="Progress"):
        k+=1
        
        batch = createImageCubes(X, input_tensor_idx[i:i + batch_size], windowSize)
        batch =  np.reshape(batch,[np.shape(batch)[0], np.shape(batch)[1]*np.shape(batch)[2]*np.shape(batch)[3]] )

        batch_predictions = model.predict(batch)
        
        predictions.append(batch_predictions)
        
    Y_pred_test = np.concatenate(predictions, axis=0)
  
    return Y_pred_test



def get_patch_bounds(row, col, patch_size, H, W):
    margin = patch_size // 2

    r1 = max(row - margin, 0)
    r2 = min(row + margin + 1, H)

    c1 = max(col - margin, 0)
    c2 = min(col + margin + 1, W)

    return r1, r2, c1, c2

def calculate_same_class_leakage(train_image, test_image, patch_size):
    """
    Calculate same-class pixel-level leakage between training and testing patches.

    Parameters
    ----------
    train_image : 2D numpy array
        Label image containing training center pixels.
        Background/unlabelled pixels should be 0.

    test_image : 2D numpy array
        Label image containing testing center pixels.
        Background/unlabelled pixels should be 0.

    patch_size : int
        Patch size, e.g., 9, 11, 13.

    Returns
    -------
    leakage_percentage : float
        Total same-class overlapped pixels divided by the total number of
        pixels inside all testing patches, multiplied by 100.

    average_overlap_ratio : float
        Average overlap ratio among testing patches that have non-zero
        same-class overlap.

    total_same_class_overlap : int
        Total number of same-class overlapped pixels.

    total_test_patch_pixels : int
        Total number of pixels inside all testing patches.

    num_overlapped_test_patches : int
        Number of testing patches with non-zero same-class overlap.
    """

    train_image = np.asarray(train_image)
    test_image = np.asarray(test_image)

    if train_image.shape != test_image.shape:
        raise ValueError("train_image and test_image must have the same shape.")

    H, W = train_image.shape

    train_centers = np.argwhere(train_image > 0)
    test_centers = np.argwhere(test_image > 0)

    class_labels = np.unique(train_image)
    class_labels = class_labels[class_labels > 0]

    # Create one training coverage map for each class.
    # Each map indicates which pixels are covered by training patches of that class.
    train_coverage_by_class = {}

    for cls in class_labels:
        train_coverage_by_class[cls] = np.zeros((H, W), dtype=bool)

    # Mark the pixels covered by each training patch in the corresponding class map.
    for row, col in train_centers:
        cls = train_image[row, col]

        r1, r2, c1, c2 = get_patch_bounds(row, col, patch_size, H, W)

        train_coverage_by_class[cls][r1:r2, c1:c2] = True

    total_same_class_overlap = 0
    total_test_patch_pixels = 0

    overlap_ratios = []

    # Check each testing patch against the training coverage map of the same class.
    for row, col in test_centers:
        test_cls = test_image[row, col]

        r1, r2, c1, c2 = get_patch_bounds(row, col, patch_size, H, W)

        test_patch_pixels = (r2 - r1) * (c2 - c1)

        if test_cls in train_coverage_by_class:
            same_class_overlap = np.sum(
                train_coverage_by_class[test_cls][r1:r2, c1:c2]
            )
        else:
            same_class_overlap = 0

        total_same_class_overlap += same_class_overlap
        total_test_patch_pixels += test_patch_pixels

        # Compute overlap ratio for this testing patch.
        # Only non-zero overlap cases are included in the average.
        if same_class_overlap > 0:
            overlap_ratio = same_class_overlap / test_patch_pixels
            overlap_ratios.append(overlap_ratio)

    leakage_percentage = (
        total_same_class_overlap / total_test_patch_pixels
    ) * 100 if total_test_patch_pixels > 0 else 0.0

    average_overlap_ratio = (
        np.mean(overlap_ratios) * 100
    ) if len(overlap_ratios) > 0 else 0.0

    num_overlapped_test_patches = len(overlap_ratios)

    return leakage_percentage, average_overlap_ratio
       


import matplotlib.patches as mpatches
def img_display2(data=None, rgb_band=None, classes=None, class_name=None, title=None, 
                figsize=(7, 7), palette=spectral.spy_colors, Location='upper left',
                unassigned_value=0):

    if data is not None:
        im_rgb = np.zeros_like(data[:, :, 0:3])
        im_rgb = data[:, :, rgb_band]
        im_rgb = im_rgb / (np.max(np.max(im_rgb, axis=1), axis=0)) * 255
        im_rgb = np.asarray(im_rgb, np.uint8)

        fig, rgbax = plt.subplots(figsize=figsize)
        rgbax.imshow(im_rgb)
        rgbax.set_title(title)
        rgbax.axis('off')
        
    elif classes is not None:
        rgb_class = np.zeros((classes.shape[0], classes.shape[1], 3))

        for i in np.unique(classes):
            if i == unassigned_value:
                rgb_class[classes == i] = [255, 255, 255]   # white
            else:
                rgb_class[classes == i] = palette[i]

        rgb_class = np.asarray(rgb_class, np.uint8)

        _, classax = plt.subplots(figsize=figsize)
        classax.imshow(rgb_class)
        classax.set_title(title)
        classax.axis('off')
        
    if class_name is not None:
        patches = []

        for i in np.unique(classes):
            if i == unassigned_value:
                color = np.array([255, 255, 255]) / 255.0
            else:
                color = np.array(palette[i]) / 255.0

            patches.append(mpatches.Patch(color=color, label=class_name[i]))

        classax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=Location,
                       borderaxespad=0., fontsize=15, title="Classes")

    plt.tight_layout()