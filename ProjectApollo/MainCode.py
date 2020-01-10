import os
import cv2
import Code
import argparse
from PIL import Image

# ImageToProcess = r"D:\mydownloads\uploads18072019\uploads\original\2"
#
# # input_img = cv2.imread(ImageToProcess)
# recognised_text = Code.MainFunction(ImageToProcess + ".JPG" ,ImageToProcess)
# print(recognised_text)

if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    #ap.add_argument("-c", "--classify", required=True, help="character and digit dataset")
    args = vars(ap.parse_args())

    # read image from disk
    imageList = sorted([x.split('.') for x in os.listdir(args["image"])], key= lambda x:int(x[0]))

    for imageName in imageList:
        if imageName[1] == "png" or imageName[1] == "jpg" or imageName[1] == "JPG" :
            Name = imageName[0]+ '.' +imageName[1]
            fullPath = os.path.join(args["image"], Name)
            fullPathwithoutext = os.path.join(args["image"], imageName[0])
            image = cv2.imread(fullPath)
            # input_img = cv2.imread(ImageToProcess)
            recognised_text = Code.MainFunction(fullPath, fullPathwithoutext)
            print(recognised_text)


