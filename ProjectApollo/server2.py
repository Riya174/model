import base64
import io
import json
import logging
import os
import sys
import time

import Code
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, Response
# count=0
from werkzeug.utils import secure_filename

# from werkzeug import secure_filename

app = Flask(__name__,)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
UPLOAD_ORG_FOLDER = '{}original/'.format(UPLOAD_FOLDER)
UPLOAD_PREP_FOLDER = '{}preprocessed/'.format(UPLOAD_FOLDER)
UPLOAD_IMG_FOLDER = './uploads'
app.config['UPLOAD_IMG_FOLDER'] = UPLOAD_IMG_FOLDER
# cpt = sum([len(files) for r, d, files in os.walk(UPLOAD_FOLDER)])
# count=cpt+1;
# print("files:",count);
app.config['UPLOAD_ORG_FOLDER'] = UPLOAD_ORG_FOLDER
app.config['UPLOAD_PREP_FOLDER'] = UPLOAD_PREP_FOLDER
# module-level variables ##############################################################################################
RETRAINED_LABELS_TXT_FILE_LOC = os.getcwd() + "/" + "retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = os.getcwd() + "/" + "retrained_graph.pb"

TEST_IMAGES_DIR = '/home/swati/Desktop/Development_3/new_dataset/test/'

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)

confthres = 0.3
nmsthres = 0.1
yolo_path = './'


def writeResultOnImage(openCVImage, resultText):
    # ToDo: this function may take some further fine-tuning to show the text well given any possible image size

    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape

    # choose a font
    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    # chose the font size and thickness as a fraction of the image size
    fontScale = 1.0
    fontThickness = 2

    # make sure font thickness is an integer, if not, the OpenCV functions that use this may crash
    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    # calculate the lower left origin of the text area based on the text area center, width, and height
    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

    # write the text on the image
    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_BLUE,
                fontThickness)


def image_classify(fileName):
    print("starting program . . .")
    # get a list of classifications from the labels file
    classifications = []
    # for each line in the label file . . .
    print(RETRAINED_LABELS_TXT_FILE_LOC)
    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        # remove the carriage return
        classification = currentLine.rstrip()
        # and append to the list
        classifications.append(classification)
    # end for

    # show the classifications to prove out that we were able to read the label file successfully
    print("classifications = " + str(classifications))

    # load the graph from file
    with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        # instantiate a GraphDef object
        graphDef = tf.GraphDef()
        # read in retrained graph into the GraphDef object
        graphDef.ParseFromString(retrainedGraphFile.read())
        # import the graph into the current default Graph, note that we don't need to be concerned with the return value
        _ = tf.import_graph_def(graphDef, name='')
    # end with

    with tf.Session() as sess:
        # if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
        # if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
        #     break
        # end if

        # show the file name on std out
        print(fileName)

        # get the file name and full path of the current image file
        imageFileWithPath = fileName
        try:
            # attempt to open the image with OpenCV
            # openCVImage = cv2.imread(imageFileWithPath)
            openCVImage = fileName
        except Exception as ex:
            print("Error while reading an image: " + str(ex))

        # if we were not able to successfully open the image, continue with the next iteration of the for loop
        if openCVImage is None:
            print("unable to open " + fileName + " as an OpenCV image")
            # break
        # end if

        # get the final tensor from the graph
        finalTensor = sess.graph.get_tensor_by_name('final_result:0')

        # convert the OpenCV image (numpy array) to a TensorFlow image
        tfImage = np.array(openCVImage)[:, :, 0:3]

        # run the network to get the predictions
        predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

        # sort predictions from most confidence to least confidence
        sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

        print("---------------------------------------")

        # keep track of if we're going through the next for loop for the first time so we can show more info about
        # the first prediction, which is the most likely prediction (they were sorted descending above)
        onMostLikelyPrediction = True
        # for each prediction . . .
        for prediction in sortedPredictions:
            strClassification = classifications[prediction]

            # if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
            if strClassification.endswith("s"):
                strClassification = strClassification[:-1]
            # end if

            # get confidence, then get confidence rounded to 2 places after the decimal
            confidence = predictions[0][prediction]

            # if we're on the first (most likely) prediction, state what the object appears to be and show a % confidence to two decimal places
            if onMostLikelyPrediction:
                # get the score as a %
                scoreAsAPercent = confidence * 100.0

                return strClassification, "{0:.2f}".format(scoreAsAPercent)
                # show the result to std out
                print("the object appears to be a " + strClassification + ", " + "{0:.2f}".format(
                    scoreAsAPercent) + "% confidence")
                # write the result on the image
                writeResultOnImage(openCVImage,
                                   strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                # finally we can show the OpenCV image
                cv2.imshow(fileName, openCVImage)
                # mark that we've show the most likely prediction at this point so the additional information in
                # this if statement does not show again for this image
                onMostLikelyPrediction = False
            # end if

            # for any prediction, show the confidence as a ratio to five decimal places
            print(strClassification + " (" + "{0:.5f}".format(confidence) + ")")
            # return strClassification + " (" + "{0:.5f}".format(confidence) + ")"
            return None


@app.route('/detectPattern', methods=['POST'])
def api_root():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:

        # file = request.files['image']
        # if file:
        #     filename = secure_filename(file.filename)
        #     print(filename)
        #     file.save(os.path.join(app.config['UPLOAD_IMG_FOLDER'], filename))

        data = request.files.to_dict()
        inner_data = data['image']
        format = str(inner_data)
        # i=format.find('image/jpeg')
        # j = format.find('image/png')
        img_data = request.files['image'].read()

        im = cv2.imdecode(np.asarray(bytearray(img_data), dtype=np.uint8), 1)
        # ext = img_data.name.split(".")
        # print(ext)
        # print(img_data)
        # image = Image.open(im)
        # image.save('/var/tmp/cmprs_' + str(im))
        # kernel = np.ones((5, 5), np.uint8)
        # print(type(img_data))
        # print('/var/tmp/cmprs_' + str(im))
        text = "test"
        # print(os.path.join(app.config['UPLOAD_IMG_FOLDER'], filename))
        return_text, percent = image_classify(im)
        print(return_text)
        # os.remove('/var/tmp/cmprs_1' + str(im))

        return Response(json.dumps({"predicted_pattern": return_text, "predicted_percentage": percent}),
                        mimetype='application/json')
    else:
        return Response(json.dumps({"predicted_text": "Please rescan VIN!"}), mimetype='application/json')


def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    # labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
    lpath = os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS


def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    return COLORS


def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath


def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath


def load_model(configpath, weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def image_to_byte_array(image: Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def get_predection(image, net, LABELS, COLORS):
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W * 2, H * 1.1])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)
    nos = []
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            text = "{}".format(LABELS[classIDs[i]])
            print(boxes)
            print(classIDs)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            nos.append(text)
    print(nos)
    return image


labelsPath = "data/obj.names"
cfgpath = "yolov3-tiny-obj.cfg"
wpath = "backup/yolo-obj_4000.weights"
Lables = get_labels(labelsPath)
CFG = get_config(cfgpath)
Weights = get_weights(wpath)
nets = load_model(CFG, Weights)
Colors = get_colors(Lables)


@app.route('/detectstensil', methods=['POST'])
def detect_stensil():
    # load our input image and grab its spatial dimensions
    # image = cv2.imread("./test1.jpg")0000
    img=request.files["image"].read()
    #img = base64.b64decode(request.form["image"])
    img = Image.open(io.BytesIO(img))
    npimg = np.array(img)

    #npimg = np.fromstring(img,np.uint8)
    image = npimg.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = get_predection(image, nets, Lables, Colors)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # show the output image
    # cv2.imshow("Image", res)
    # cv2.waitKey()
    image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    np_img = Image.fromarray(image)
    img_encoded = image_to_byte_array(np_img)
    cv2.imwrite("image.jpg", res)
    data={}
    with open("image.jpg" , mode="rb") as file:
        img2=file.read()
    data["img2"]=base64.encodebytes(img2).decode("utf-8")

    return Response(response=json.dumps(data), status=200, mimetype="application/json")


@app.route('/detectDefect',methods= ['POST'])
def api_root2():
    app.logger.info(PROJECT_HOME)
    if request.method=='POST' and request.files['image']:
        ROOT_DIR = os.getcwd()

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        from mrcnn import utils
        from mrcnn import visualize
        import mrcnn.model as modellib
        from mrcnn.model import log

        import custom

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        custom_WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_damage_0050.h5")  # TODO: update this path

        config = custom.CustomConfig()
        #custom_DIR = r"D:\RIYA\pattern_classification\pattern_classification\dataset"

        # Override the training configurations with a few
        # changes for inferencing.
        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        #config.display()

        # Device to load the neural network on.
        # Useful if you're training a model on the same
        # machine, in which case use CPU and leave the
        # GPU for training.
        DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

        # Inspect the model in training or inference modes
        # values: 'inference' or 'training'
        # TODO: code for 'training' test mode not ready yet
        TEST_MODE = "inference"

        def get_ax(rows=1, cols=1, size=16):
            """Return a Matplotlib Axes array to be used in
            all visualizations in the notebook. Provide a
            central point to control graph sizes.

            Adjust the size attribute to control how big to render images
            """
            _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
            return ax

        # Load validation dataset
        # dataset = custom.CustomDataset()
        # dataset.load_custom(custom_DIR, "val")

        # Must call before using the dataset
        # dataset.prepare()

        # print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                      config=config)

        # load the last model you trained
        # weights_path = model.find_last()[1]

        # Load weights
        print("Loading weights ", custom_WEIGHTS_PATH)

        model.load_weights(custom_WEIGHTS_PATH, by_name=True)

        from importlib import reload  # was constantly changin the visualization, so I decided to reload it instead of notebook
        reload(visualize)

        #import argparse
        # Parse command line arguments
        #parser = argparse.ArgumentParser(
            #description='Train Mask R-CNN to detect custom class.')
        #parser.add_argument('--image', required=False,
                            #metavar="path or URL to image",
                            #help='Image to apply the color splash effect on')
        #args = parser.parse_args()
        # image_id = random.choice(dataset.image_ids)
        # image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        # modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        # info = dataset.image_info[image_id]
        # print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
        # dataset.image_reference(image_id))

        # image = skimage.io.imread(args.image)
        image1 = request.files["image"].read()


        image2 = Image.open(io.BytesIO(image1))
        image = np.array(image2)
        # Run object detection

        results = model.detect([image], verbose=1)

        # Display results
        ax = get_ax(1)
        r = results[0]

        capt = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                           ['BG', 'deflation_damage', 'side_through_cut', 'impact_bulge'], r['scores'],
                                           ax=ax,
                                           title="Predictions")

        data={}
        with open("image.jpg", mode='rb') as file:
            img = file.read()
        data['img'] = base64.encodebytes(img).decode('utf-8')
        data['cap'] = capt

        return Response(response=json.dumps(data), status=200, mimetype='application/json')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def create_new_folder(local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    return local_dir
@app.route('/detectText', methods = ['POST'])
def api_root4():
    app.logger.info(PROJECT_HOME)
    app.logger.info(UPLOAD_FOLDER)
    if request.method == 'POST' and request.files['image']:

        data = request.files.to_dict()
        inner_data = data['image']
        format = str(inner_data)
        i = format.find('image/jpeg')
        j = format.find('image/png')
        img = request.files['image'].read()
        size = len(img)
        cpt = sum([len(files) for r, d, files in os.walk(UPLOAD_FOLDER)])
        count = cpt + 1

        #if i != -1 or j != -1:
        #if size <= 1048576:
        img = request.files['image']
        original_img = Image.open(img)
        # resized_img = original_img.resize((320, 320))
        img_name = secure_filename(img.filename)
        #print("image name:", img_name)
        #output = ocr_core(img_name)
        sub = img_name.rfind('.')
        length = len(img_name)
        substr = img_name[sub:length]


        create_new_folder(app.config['UPLOAD_FOLDER'])
        count = cpt + 1
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], str(count) + substr)

        app.logger.info("saving {}".format(saved_path))
        original_img.save(saved_path)
        #print(saved_path)
        recognised_text = Code.MainFunction(saved_path)
        if "M" in recognised_text and len(recognised_text.split("MR", 1)) >1:
            recognised_text = 'MR'+ recognised_text.split('MR', 1)[1]
            recognised_text = recognised_text[0:17]
        if "B" in recognised_text and len(recognised_text.split("BL", 1)) >1:
            recognised_text = 'BL'+ recognised_text.split('BL', 1)[1]
            recognised_text = recognised_text[0:17]

        if len(recognised_text) > 17: recognised_text = recognised_text[:-1]
        if len(recognised_text) < 15 or  '-' in recognised_text: recognised_text = recognised_text[1:11]
        if(recognised_text[9] in ['h', '1']):
            recognised_text.replace(recognised_text[9], 'l')
        recognised_text = recognised_text.upper()
        print(recognised_text)
        #if(recognised_text[9] in ['h', '1']):
        #    recognised_text.replace(recognised_text[9], 'l')
        if recognised_text is not None and len(recognised_text) > 0 :
            return Response(json.dumps({"predicted_text": recognised_text}), mimetype='application/json')
        else:   return Response(json.dumps({"predicted_text":recognised_text}), mimetype='application/json')
            #else:
            #    return Response(json.dumps({"predicted_text": "Size should not be greater than 1 mb"}),mimetype='application/json')
        #else:
        #    return Response(json.dumps({"predicted_text": "Size should not be greater than 1 mb"}), mimetype='application/json')
    else:
        return Response(json.dumps({"predicted_text": "Where is the image?"}),mimetype='application/json')

@app.route('/detectDamage', methods = ['POST'])
def api_root3():
    app.logger.info(PROJECT_HOME)
    if request.method=='POST' and request.files['image']:
        ROOT_DIR = os.getcwd()

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        from mrcnn import utils
        from mrcnn import visualize_car
        from mrcnn.visualize import display_images
        import mrcnn.model as modellib
        from mrcnn.model import log

        import custom_car

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        custom_WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_damage_0020.h5")  # TODO: update this path

        config = custom_car.CustomConfig()
        #custom_DIR = r"D:\RIYA\pattern_classification\pattern_classification\dataset_dir"

        # Override the training configurations with a few
        # changes for inferencing.
        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        # config.display()

        # Device to load the neural network on.
        # Useful if you're training a model on the same
        # machine, in which case use CPU and leave the
        # GPU for training.
        DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

        # Inspect the model in training or inference modes
        # values: 'inference' or 'training'
        # TODO: code for 'training' test mode not ready yet
        TEST_MODE = "inference"

        def get_ax(rows=1, cols=1, size=16):
            """Return a Matplotlib Axes array to be used in
            all visualizations in the notebook. Provide a
            central point to control graph sizes.

            Adjust the size attribute to control how big to render images
            """
            _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
            return ax

        # Load validation dataset
        # dataset = custom_car.CustomDataset()
        # dataset.load_custom(custom_DIR, "val")

        # Must call before using the dataset
        #dataset.prepare()



        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                      config=config)

        # load the last model you trained
        # weights_path = model.find_last()[1]

        # Load weights
        print("Loading weights ", custom_WEIGHTS_PATH)
        model.load_weights(custom_WEIGHTS_PATH, by_name=True)

        from importlib import reload  # was constantly changin the visualization, so I decided to reload it instead of notebook
        reload(visualize_car)

        # image_id = random.choice(dataset.image_ids)
        # image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        #     modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        # info = dataset.image_info[image_id]
        # print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
        #                                        dataset.image_reference(image_id)))
        image1 = request.files['image'].read()
        image2 = Image.open(io.BytesIO(image1))

        # Run object detection
        image = np.array(image2)
        results = model.detect([image], verbose=1)

        # Display results
        ax = get_ax(1)
        r = results[0]
        capt = visualize_car.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    ['BG', 'damage'], r['scores'], ax=ax,
                                    title="Predictions")

        data={}
        with open("image2.jpg", mode='rb') as file:
            img = file.read()
        data['img'] = base64.encodebytes(img).decode("utf-8")
        data['cap'] = capt

        return Response(response=json.dumps(data), status=200, mimetype='application/json')



if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
