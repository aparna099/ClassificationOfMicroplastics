import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os
from scipy.ndimage import rotate

import albumentations as A
images_to_generate=2




images_path="C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/images" #path to original images
masks_path = "C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/masks"
img_augmented_path="C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/aug1" # path to store aumented images
msk_augmented_path="C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/aug1mask" # path to store aumented images
images=[] # to store paths of images from folder
masks=[]

for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     
    images.append(os.path.join(images_path,im))

for msk in os.listdir(masks_path):  # read image name from folder and append its path into "images" array     
    masks.append(os.path.join(masks_path,msk))


aug = A.Compose([
    A.VerticalFlip(p=1),              
    A.CenterCrop(p=1, height=2048, width=2560),
    A.HorizontalFlip(p=1),
    A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.GridDistortion(p=1),
    A.OpticalDistortion(distort_limit=0.5, shift_limit=1, p=1)
    ]
)

#random.seed(42)

i=1   # variable to iterate till images_to_generate


while i<=images_to_generate: 
    #number = random.randint(0, len(images)-1)  #PIck a number to select an image & mask
    number = 0
    image = images[number]
    mask = masks[number]
    print(image, mask)
    #image=random.choice(images) #Randomly select an image name
    original_image = io.imread(image)
    original_mask = io.imread(mask)
    
    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']

        
    new_image_path= "%s/%s.jpg" %(img_augmented_path, i)
    new_mask_path = "%s/%s.jpg" %(msk_augmented_path, i)
    io.imsave(new_image_path, transformed_image)
    io.imsave(new_mask_path, transformed_mask)
    i =i+1