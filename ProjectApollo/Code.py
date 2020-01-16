import numpy as np
import tensorflow as tf
import cv2 as cv
import imutils
from operator import itemgetter

import CharacterRecog as CRecog
import Textboxes as Tbox
import os
import shutil
import cv2
import tensorflow as tf
import csv
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.preprocessing.image import img_to_array
import pickle
import string, random
from PIL import Image
from hubmaster.examples.image_retraining import label_image
# import ray
# ray.init()

# Read the graph.

search_path = 'Uploads/'
save_path = 'TEMP/'
extensions = ['.jpg','.JPG','.png','.jpeg']
imgnames = []

# @ray.remote
def predictionLabel(i, save_path, image_name):
    Name = str(i) + '.jpg'
    Name = image_name + '-' + Name
    fullPath = os.path.join(save_path, image_name, 'Characters', Name)
    ROOT_DIR = os.getcwd()
    prediction = label_image.predict_label(fullPath,
                                           os.path.join(ROOT_DIR,"digit_output_graph.pb"),
                                           os.path.join(ROOT_DIR,"digit_output_labels"),
                                          0, 0, 0, 0, "Placeholder", "final_result")
    print("prediction ", prediction)
    return prediction


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def MainFunction(filename):
    print(filename)

    if os.path.splitext(os.path.basename(filename))[1] in extensions:
        image_name = os.path.splitext(os.path.basename(filename))[0]
        input_img = cv2.imread(filename)
        try:
            os.mkdir(save_path + image_name)
        except:
            # yn = raw_input('Directory ' + image_name + ' already exists. Overwrite? (y/n): ')
            yn = 'Y'
            if yn=='y' or yn=='Y':
                shutil.rmtree(save_path + image_name)
                os.mkdir(save_path + image_name)
            else:
                print("CONTINUE")
        try:
            os.mkdir(save_path + image_name + '/Characters')
        except:
            print('Error encountered in creating Characters directory at' + image_name + '. Skipping Image')

        print('Processing image ' + image_name + '. Please wait...')
        #textbox_rect = Tbox.TextboxDetector(input_img)
        #cv2.imwrite("textbox_rect.JPG", textbox_rect)
        augmented = Tbox.Augmenting(input_img)
        #cv2.imwrite("augmented.JPG", augmented)

        #TODO: SET GAMMA CORRECTION AND ALPHA VALUES TO A REASONABLE AMOUNT. THIS MAY DEPEND ON IMAGE AND WILL HELP TO IMPROVE MODEL ACCURACY
        gamma = 2               #SET TO 1 TO MAKE NO CHANGES TO IMAGE
        alpha = 1.5             #SET TO 1 TO MAKE NO CHANGES TO IMAGE
        doSmoothing = 1         #SET TO 0 TO MAKE NO CHANGES TO IMAGE
        kernel_size = 15        #KERNEL SIZE FOR SMOOTHING
        smoothing_param = 80    #SIGMA VALUE FOR SMOOTHING
        VIN = ""
        # improved_img = Tbox.ImageImprov(augmented, gamma, alpha, doSmoothing, kernel_size, smoothing_param)
        improved_img = augmented
        CRecog.CharDetection(improved_img, image_name = image_name, save_path = save_path)
        lenImg = len(os.listdir(save_path + image_name + '/Characters/'))
        futures = [predictionLabel(i, save_path, image_name) for i in range(lenImg)]
        return "".join(futures)
        # for i in range(len(os.listdir(save_path + image_name + '/Characters/'))):
        #     Name = str(i) + '.jpg'
        #     Name = image_name + '-' + Name
        #     fullPath = os.path.join(save_path, image_name, 'Characters', Name)
        #     # characimg = image.load_img(save_path + image_name + '/Characters/' + image_name + '-' + str(i) + '.jpg',
        #     #                           target_size=(50, 100))
        #
        #     prediction = label_image.predict_label(fullPath, r"D:\TensorflowAPI-master\digit_output_graph.pb",
        #                                            r"D:\TensorflowAPI-master\digit_output_labels",
        #                                            0, 0, 0, 0, "Placeholder", "final_result")
        #
        #     print(prediction)

            # characimg1 = np.array(characimg)
            # output = imutils.resize(characimg1, width=400)
            #
            # # pre-process the image for classification
            # characimg1 = cv2.resize(characimg1, (96, 96))
            # characimg3 = Image.fromarray(characimg1)
            # characimg2 = characimg1.astype("float") / 255.0
            # characimg2 = img_to_array(characimg2)
            # characimg2 = np.expand_dims(characimg2, axis=0)
            #
            # # load the trained convolutional neural network and the multi-label
            # # binarizer
            # print("[INFO] loading network...")
            # model = load_model(r"D:\mydownloads\keras-multi-label\keras-multi-label\keras1.model")
            # mlb = pickle.loads(open(r"D:\mydownloads\keras-multi-label\keras-multi-label\mlb11.pickle", "rb").read())
            #
            # # classify the input image then find the indexes of the two class
            # # labels with the *largest* probability
            # print("[INFO] classifying image...")
            # proba = model.predict(characimg2)[0]
            # idxs = np.argsort(proba)[::-1][:2]
            #
            # # loop over the indexes of the high confidence class labels

            # for (i, j) in enumerate(idxs):
            #     # build the label and draw the label on the image
            #     label =mlb.classes_[j]
            # print(label)
            # imgName = prediction + "_"
            # imgName = imgName + randomString()
            # imgName = imgName + ".JPG"
            # shutil.copyfile(fullPath, os.path.join(r"C:\Users\pnn916765\Downloads\Harshal\Digits", prediction, imgName))

        # recognised_text_with_Confidence = CRecog.CharRecognition(image_name = image_name,
        # save_path = save_path)
        # recognised_text_SecondModel = CRecog.CharRecognitionSecondModel(image_name = image_name, save_path = save_path)
        # print(recognised_text_SecondModel)
        # recognised_text = []
        # for i in range (0, len(recognised_text_with_Confidence)):
        #    if float(recognised_text_with_Confidence[i][1]) > 65:
        #        recognised_text.append(recognised_text_with_Confidence[i][0])
        #    else:
        #        recognised_text.append(recognised_text_SecondModel[i])
        #
        # return recognised_text

        #automate image classification

        return "APPLIED BOUNDING BOX"

