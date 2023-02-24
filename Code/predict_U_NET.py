import os
import cv2
import pandas
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.io import imread
from skimage import measure, color
from U_NET_model import U_NET_model
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify


image_path = 'prediction/images'

image_ids = os.listdir(image_path)

#Get Unet model

def get_U_NET_model():
        return U_NET_model(256, 256, 3)

model = get_U_NET_model()

model.load_weights('MICROPLASTIC_MODEL.hdf5')

print("Reading Image for prediction")

for n, image in tqdm(enumerate(image_ids), total=len(image_ids)):
    
    im= Image.open(os.path.join(image_path,image))
    im = im.resize((2560,2048))
    im.save(os.path.join(image_path,image))
    
    large_image = imread(os.path.join(image_path,image))[:,:,:3]
    
    im= Image.open(os.path.join(image_path,image))
    im = im.resize((2592,1944))
    im.save(os.path.join(image_path,image))

    #This will split the image into small images of shape [3,3]

    patches = patchify(large_image, (256, 256, 3), step=128)  #Step=256 for 256 patches means no overlap
    patches = patches.squeeze()
    predicted_patches = []

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
        
            single_patch = patches[i,j,:,:,:]
            single_patch_norm = np.float32((np.array(single_patch) / 255.))
            single_patch_input=np.expand_dims(single_patch_norm, 0)

            #Predict and threshold for values above 0.01 probability
            predict = model.predict(single_patch_input)[0,:,:,:]
            single_patch_prediction =  (predict > 1e-3).astype(np.uint8)
            predicted_patches.append(single_patch_prediction)

    predicted_patches = np.array(predicted_patches)

    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256, 256))
    reconstructed_image = unpatchify(predicted_patches_reshaped, (large_image.shape[0], large_image.shape[1]))
    plt.imshow(np.squeeze(reconstructed_image), cmap='gray')
    plt.show()
    plt.imsave('prediction/semantic/' + image, reconstructed_image, cmap='gray')
    sem_path = 'prediction/semantic/' + image
    im= Image.open(sem_path)
    im = im.resize((2592,1944))
    im.save(sem_path)

    #plotting the orginal and predicted image

    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('Input Image')
    plt.imshow(large_image, cmap='gray')
    plt.subplot(222)
    plt.title('Predicted Image')
    plt.imshow(np.squeeze(reconstructed_image), cmap='gray')
    plt.show()


    ##################################
    #Watershed to convert semantic to instance
    #########################

    img = cv2.imread('prediction/semantic/' + image)  #Read as RBB image
    img_grey = img[:,:,0]
    
    ret1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 0)
   
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
   
    sure_fg = cv2.erode(opening,kernel,iterations = 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 10

    markers[unknown==255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0,255,255]  


    img2 = color.label2rgb(markers, bg_label=10, bg_color = (1,1,1))
    
    np.savetxt('prediction/location/' + image[:-4] + '.csv', markers, fmt = "%d", delimiter = ',')
    plt.imsave('prediction/instance/' + image, img2)
    
    '''
    cv2.namedWindow('Colored Grains', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Colored Grains', img2)
    cv2.resizeWindow('Colored Grains', 1280,768)
    cv2.waitKey(0)
    '''

    propList = ['Label', 'Area','Perimeter', 'equivalent_diameter']

    regions = measure.regionprops(markers, intensity_image=img_grey)
                            
    
    output_file = open('prediction/measurements/' + image[:-4] + '.csv', 'w')
    output_file.write('Microplastic No.' + "," + ",".join(propList) + '\n')  #join strings in array by commas

    particle_number = 1
    pixels_to_um = 0.5
    for region_props in regions:
        
        #output cluster properties to the excel file
        output_file.write(str(particle_number))
        
        for i,prop in enumerate(propList):
            
            if(prop == 'Area'): 
                #Convert pixel square to um square
                to_print = region_props[prop] * pixels_to_um**2  

            elif(prop == 'equivalent_diameter'): 
                to_print = region_props[prop] * pixels_to_um  
            else:
                #Remaining props 
                to_print = region_props[prop]     
            output_file.write(',' + str(to_print))
        output_file.write('\n')
        particle_number += 1
    
    output_file.close()#Closes the file, otherwise it would be read only. 
    
    
    eq_diameter = pandas.read_csv('prediction/measurements/' + image[:-4] + '.csv', usecols=["equivalent_diameter"])

    diameter_ranges = eq_diameter.equivalent_diameter.tolist()
    diameter_ranges = diameter_ranges[1:]

    plt.hist(diameter_ranges, bins = [0,2,5,10,30,50,150])
    plt.xlabel('Equivalent Diameter (micrometer)')
    plt.ylabel('No. of Microplastics')
    plt.title('Size Distribution of Microplastics')
    plt.savefig("prediction/size_dist/" + image)
    


    
