from flask import Flask, request, Response
import logging, os
import json
import io, os, shutil, time, random
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
from werkzeug import secure_filename

# count=0
app = Flask(__name__)
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
    if request.method == 'POST' and request.files['Image']:

        # file = request.files['Image']
        # if file:
        #     filename = secure_filename(file.filename)
        #     print(filename)
        #     file.save(os.path.join(app.config['UPLOAD_IMG_FOLDER'], filename))

        data = request.files.to_dict()
        inner_data = data['Image']
        format = str(inner_data)
        # i=format.find('image/jpeg')
        # j = format.find('image/png')
        img_data = request.files['Image'].read()

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
    # image = cv2.imread("./test1.jpg")

    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    npimg = np.array(img)
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
    return Response(response=img_encoded, status=200, mimetype="image/jpeg")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
