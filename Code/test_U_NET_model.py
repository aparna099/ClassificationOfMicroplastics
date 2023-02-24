import os
import random
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from U_NET_model import U_NET_model
from matplotlib import pyplot as plt



IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

image_directory = 'stage1_test/images/'
mask_directory = 'stage1_test/masks/'

train_ids = os.listdir(image_directory)
train_ids.sort()

mask_ids = os.listdir(mask_directory)
mask_ids.sort()

X_test = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

print('Resizing testing images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = os.path.join(image_directory,id_)
    img = imread(path)[:,:,:IMG_CHANNELS]  
    X_test[n] = img  #Fill empty X_train with values from img

for n, id_ in tqdm(enumerate(mask_ids), total=len(mask_ids)):   
    path = os.path.join(mask_directory,id_)
    msk = imread(path)[:,:,:1] 
    Y_test[n] = msk  #Fill empty Y_train with values from mask
    
print('Done!')


image = X_test
mask = Y_test

X_test = (np.array(X_test)) / 255.
X_test = X_test.astype(np.float32)

Y_test = (np.array(Y_test)) /255.
Y_test = Y_test.astype(np.float32)


def get_U_NET_model():
    return U_NET_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model_jaccard = get_U_NET_model()

model_jaccard.load_weights('MICROPLASTIC_MODEL_overall.hdf5')


#IOU
y_pred = model_jaccard.predict(X_test, batch_size = 32, verbose = 1)
y_pred_thresholded = y_pred > 1e-3

intersection = np.logical_and(Y_test , y_pred_thresholded)
union = np.logical_or(Y_test , y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre = ", iou_score)

#Accuracy on test images

_, acc = model_jaccard.evaluate(X_test, Y_test, verbose = 1)
print("Accuracy of U-Net Model = ", (acc * 100.0), "%")


#predicting on testing dataset  

preds_val = model_jaccard.predict(X_test, batch_size = 32, verbose=1)
preds_val_t = (preds_val > 1e-3).astype(np.uint8)

# Perform a sanity check on some random testing samples

ix = random.randint(0, len(preds_val_t))
plt.figure(figsize=(10, 8))
plt.subplot(231)
plt.title('Test Image')
plt.imshow(image[ix])
plt.subplot(232)
plt.title('Test Mask')
plt.imshow(np.squeeze(mask[ix]), cmap ='gray')
plt.subplot(233)
plt.title('Predicted image')
plt.imshow(np.squeeze(preds_val_t[ix]), cmap ='gray')
plt.show()