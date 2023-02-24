import os
  
# Function to rename multiple files
def main():
    count = 1
    path = 'C:/Users/George Devassy/Desktop/FAMILY/JITHIN DETAILS/Final Project/code/trainingdataset'
    files = os.listdir(path)
    for filename in files:
        dst = str(count) + '.jpg'
        src =os.path.join(path, filename)
        dst =os.path.join(path, dst)
        count = count + 1  
        # rename() function will
        # rename all the files
        os.rename(src, dst)
  
# Driver Code
if __name__ == '__main__':
      
    # Calling main() function
    main()