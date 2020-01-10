import CharacterRecog as CRecog
import Textboxes as Tbox
import os
import shutil
import cv2
import tensorflow as tf
import csv


search_path = 'Uploads/'
save_path = 'TEMP/'
extensions = ['.jpg','.JPG','.png','.jpeg']
imgnames = []

if search_path != 'Uploads/':
    print('You have set the search path outside the recommended folder! Image overwriting may occur.')
if len(os.listdir(save_path)):
    print('Save Path already has some content, overwriting may occur.')

for filename in os.listdir(search_path):
    filepath = search_path + filename
    
    if os.path.splitext(os.path.basename(filepath))[1] in extensions:
        image_name = os.path.splitext(os.path.basename(filepath))[0]
        input_img = cv2.imread(filepath)
        try:
            os.mkdir(save_path + image_name)
        except:
            yn = raw_input('Directory ' + image_name + ' already exists. Overwrite? (y/n): ')
            if yn=='y' or yn=='Y':
                shutil.rmtree(save_path + image_name)
                os.mkdir(save_path + image_name)
            else: 
                continue
        try:
            os.mkdir(save_path + image_name + '/Characters')
        except:
            print('Error encountered in creating Characters directory at' + image_name + '. Skipping Image')

        print('Processing image ' + image_name + '. Please wait...')
        textbox_rect = Tbox.TextboxDetector(input_img)
        augmented = Tbox.Augmenting(textbox_rect)

        #TODO: SET GAMMA CORRECTION AND ALPHA VALUES TO A REASONABLE AMOUNT. THIS MAY DEPEND ON IMAGE AND WILL HELP TO IMPROVE MODEL ACCURACY
        gamma = 2               #SET TO 1 TO MAKE NO CHANGES TO IMAGE
        alpha = 1.5             #SET TO 1 TO MAKE NO CHANGES TO IMAGE
        doSmoothing = 1         #SET TO 0 TO MAKE NO CHANGES TO IMAGE
        kernel_size = 15        #KERNEL SIZE FOR SMOOTHING
        smoothing_param = 80    #SIGMA VALUE FOR SMOOTHING

        improved_img = Tbox.ImageImprov(augmented, gamma, alpha, doSmoothing, kernel_size, smoothing_param)
        CRecog.CharDetection(improved_img, image_name = image_name, save_path = save_path)
        recognised_text = CRecog.CharRecognition(image_name = image_name, save_path = save_path)
        imgnames.append([image_name, recognised_text])

print(imgnames)
with open('Output.csv','w') as f:
    wr = csv.writer(f)
    wr.writerows(imgnames)