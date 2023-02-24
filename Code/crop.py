#Import required Image library
from PIL import Image
#Create an Image Object from an Image
path1 = 'C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/aug1/5.jpg'
path2 = 'C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/aug1mask/5.jpg'
im1 = Image.open(path1)
im2 = Image.open(path2)
im1 = im1.resize((2560,2048))
im2 = im2.resize((2560,2048))
#left, upper, right, lowe
#Crop
left = 0
top = 0
right = 256
bottom = 256
count = 880
for i in range(8):
    for j in range(10):
        count = count + 1
        crop1 = im1.crop((left,top,right,bottom))
        crop2 = im2.crop((left,top,right,bottom))
        left = left + 256
        right = right + 256
        new_path1 = "C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/"+"stage1_train/images/img"+str(count)+".jpg"
        new_path2 = "C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/"+"stage1_train/masks/mask"+str(count)+".jpg"
        crop1.save(new_path1)
        crop2.save(new_path2)
    left = 0
    right = 256
    top = top + 256
    bottom = bottom + 256
