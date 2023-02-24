import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from skimage.io import imread
from U_NET_model import U_NET_model 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

image_directory = 'stage1_train/images/'
mask_directory = 'stage1_train/masks/'

train_ids = os.listdir(image_directory)
train_ids.sort()

mask_ids = os.listdir(mask_directory)
mask_ids.sort()

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(mask_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)


print('Reading training images and masks')

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = os.path.join(image_directory,id_)
    img = imread(path)[:,:,:IMG_CHANNELS]  
    X_train[n] = img  #Fill empty X_train with values from img

for n, id_ in tqdm(enumerate(mask_ids), total=len(mask_ids)):   
    path = os.path.join(mask_directory,id_)
    msk = imread(path)[:,:,:1] 
    Y_train[n] = msk  #Fill empty Y_train with values from mask
    
print('Done!')

    
image = X_train
mask = Y_train

#before normalizing values split the dataset to plot images 80:20
xtrain, xtest, ytrain, ytest = train_test_split(image, mask, test_size = 0.2, random_state = 0, shuffle=False)

#random checking of images
'''
image_x = random.randint(0, len(train_ids))
imshow(image[image_x])
plt.show()
imshow(mask[image_x], cmap = 'gray')
plt.show()   
'''

#normalize pixel values
#Do not normalize masks, just rescale between 0 to 1.
X_train = (np.array(X_train)) / 255.
X_train = X_train.astype(np.float32)

Y_train = (np.array(Y_train)) /255.
Y_train = Y_train.astype(np.float32)
#splitting dataset into 80% training and 20% validation 

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 0, shuffle=False)


def get_U_NET_model():
    return U_NET_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model_jaccard = get_U_NET_model()

#checkpointer


callbacks = [tf.keras.callbacks.ModelCheckpoint('MICROPLASTIC_MODEL.hdf5', verbose=1, save_best_only=True)]

history_jaccard = model_jaccard.fit(Xtrain, Ytrain, 
                                    validation_data = (Xtest,Ytest), 
                                    batch_size = 16,
                                    verbose = 1,
                                    epochs=20,
                                    shuffle = False,
                                    callbacks = callbacks)

model_jaccard.save('MICROPLASTIC_MODEL_overall.hdf5')

#Evaluate the model

_, acc = model_jaccard.evaluate(Xtest, Ytest)
print("Accuracy of Model = ", (acc * 100.0), "%")

#plot the training and validation loss at each epoch

loss = history_jaccard.history['loss']
val_loss = history_jaccard.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plot the training and validation accuracy at each epoch

jc = history_jaccard.history['jaccard_coef']
val_jc = history_jaccard.history['val_jaccard_coef']

plt.plot(epochs, jc, 'y', label='Training Jaccard Coeff.')
plt.plot(epochs, val_jc, 'r', label='Validation Jaccard Coeff.')
plt.title('Training and Validation Jaccard Coeff')
plt.xlabel('Epochs')
plt.ylabel('Jaccard Coefficient')
plt.legend()
plt.show()