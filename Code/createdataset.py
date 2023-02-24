from PIL import Image
import random, os
file = []
count = 0
images = 20
path = 'C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/images'
for _ in range(images):
    file = []
    for i in range(8):
        random_filename = random.choice([x for x in os.listdir(path)if os.path.isfile(os.path.join(path, x))])
        file.append(random_filename)

    count = count + 1
    # Opens a image in RGB mode
    im1 = Image.open('C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/images/' + file[0])
    im2 = Image.open('C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/images/' + file[1])
    im3 = Image.open('C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/images/' + file[2])
    im4 = Image.open('C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/images/' + file[3])
    im5 = Image.open('C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/images/' + file[4])
    im6 = Image.open('C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/images/' + file[5])
    im7 = Image.open('C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/images/' + file[6])
    im8 = Image.open('C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/images/' + file[7])

    #set coordinates  
    left = [0,648,1296,1944,0,648,1296,1944]
    top = [0,0,0,0,972,972,972,972]
    right = [648,1296,1944,2592,648,1296,1944,2592]
    bottom = [972,972,972,972,1944,1944,1944,1944]

    #crop images 
    crop1 = im1.crop((left[0], top[0], right[0], bottom[0]))
    crop2 = im2.crop((left[1], top[1], right[1], bottom[1]))
    crop3 = im3.crop((left[2], top[2], right[2], bottom[2]))
    crop4 = im4.crop((left[3], top[3], right[3], bottom[3]))
    crop5 = im5.crop((left[4], top[4], right[4], bottom[4]))
    crop6 = im6.crop((left[5], top[5], right[5], bottom[5]))
    crop7 = im7.crop((left[6], top[6], right[6], bottom[6]))
    crop8 = im8.crop((left[7], top[7], right[7], bottom[7]))  

    #merge 4 images
    #size = im1.size
    newpath = 'C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/images/augment/new'+ str(count) + '.jpg'
    new_im = Image.new('RGB', (2592,1944), (250,250,250))
    new_im.paste(crop1, (0,0))
    new_im.paste(crop2, (648 , 0))
    new_im.paste(crop3, (1296 , 0))
    new_im.paste(crop4, (1944,0))
    new_im.paste(crop5, (0,972))
    new_im.paste(crop6, (648,972))
    new_im.paste(crop7, (1296,972))
    new_im.paste(crop8, (1944,972))
    new_im.save(newpath,'JPEG')