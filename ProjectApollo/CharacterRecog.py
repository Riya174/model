def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # USING TO DISABLE TENSORFLOW MESSAGES
import tensorflow as tf
import cv2
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import loader

# import CharacterRecog as CRecog
import string, random


def CharDetection(img, image_name='Output', save_path='TEMP/'):
    with tf.gfile.Open('frozen_inference_graph_CharacterDet.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.
        # img = cv2.imread(filename)
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv2.resize(img, (400, 400))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        num_detections = int(out[0][0])
        # print("detection_scores:::",out)

        imgRect = img.copy()

        final_boxes = []
        for i in range(num_detections):
            if i < 17:
                classId = int(out[3][0][i])
                # print(classId)
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                final_boxes.append([int(x), int(y), int(right), int(bottom)])
                cv2.rectangle(imgRect, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
        final_boxes.sort(key=lambda x: x[0])

        q = 0
        for bp in final_boxes:
            charac = img[bp[1]:bp[3], bp[0]:bp[2]]
            cv2.imwrite(save_path + image_name + '/Characters/' + image_name + '-' + str(q) + '.jpg', charac)
            q += 1
    sess.close()
    tf.reset_default_graph()
    cv2.imwrite(save_path + image_name + '/' + image_name + '.jpg', imgRect)
    return len(final_boxes)


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def CharRecognition(image_name='Output', save_path='TEMP/'):
    # model = load_model('cnn_model_characrecog5.h5')
    # train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2)
    # training_set = train_datagen.flow_from_directory('CharacterRecognitionDataSet/train', target_size=(50, 100), batch_size=1, class_mode='categorical')
    # label_map = (training_set.class_indices)
    # print(label_map)
    final_output = []
    graph1 = load_graph('output_graph.pb')
    for i in range(len(os.listdir(save_path + image_name + '/Characters/'))):
        final_output.append(
            loader.MainFunction(graph1, save_path + image_name + '/Characters/' + image_name + '-' + str(i) + '.jpg'))
    # return ''.join(final_output)
    return final_output


def CharRecognitionSecondModel(image_name='Output', save_path='TEMP/'):
    model = load_model('cnn_model_characrecog5.h5')
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2)
    training_set = train_datagen.flow_from_directory('CharacterRecognitionDataSet/train', target_size=(50, 100),
                                                     batch_size=1, class_mode='categorical')
    label_map = (training_set.class_indices)
    # print(label_map)
    final_output = []
    for i in range(len(os.listdir(save_path + image_name + '/Characters/'))):
        characimg = image.load_img(save_path + image_name + '/Characters/' + image_name + '-' + str(i) + '.jpg',
                                   target_size=(50, 100))
        arrimg = image.img_to_array(characimg)
        arrimg = np.expand_dims(arrimg, axis=0)

        images = np.vstack([arrimg])
        classes = model.predict_classes(images, batch_size=1)
        #################################TO PREDICT CLASS PROBABILITIES###############################
        # test_image_luna = image.load_img(save_path + image_name + '/Characters/' + image_name + '-' + str(i) + '.jpg', target_size= (50, 100))
        # test_image2 = image.img_to_array(test_image_luna)/255.
        # test_image2 = np.expand_dims(test_image2, axis=0)
        # luna = model.predict_proba(test_image2)
        # print("***",max(luna[0]))
        # print("***",sum(luna[0]))

        # print(max(luna[0])/sum(luna[0]))
        final_output.append(list(label_map.keys())[list(label_map.values()).index(classes[0])])
    K.clear_session()
    return ''.join(final_output)
