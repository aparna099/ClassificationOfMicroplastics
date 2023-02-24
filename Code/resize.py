from PIL import Image
import os
#Create an Image Object from an Image
path = 'C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/images'
files = os.listdir(path)
for name in files:
    im = Image.open(os.path.join(path,name))
    im = im.resize((2560,2048))
    im.save(os.path.join(path,name))