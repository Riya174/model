import base64
import io
import json
import cv2
import sys
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
from PIL import Image
from flask import Flask, request, Response
import logging, os
import glob


app = Flask(__name__,)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/detectDefect',methods= ['POST'])
def api_root():
    app.logger.info(PROJECT_HOME)
    if request.method=='POST' and request.files['image']:
        ROOT_DIR = os.getcwd()

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        from mrcnn import utils
        from mrcnn import visualize
        from mrcnn.visualize import display_images
        import mrcnn.model as modellib
        from mrcnn.model import log

        import custom

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        custom_WEIGHTS_PATH = r"D:\Tyre_defect\logs\mask_rcnn_damage_0010.h5"  # TODO: update this path

        config = custom.CustomConfig()
        custom_DIR = r"D:\Tyre_defect\dataset"

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

        from importlib import \
            reload  # was constantly changin the visualization, so I decided to reload it instead of notebook
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

if __name__ == '__main__':
    app.run(host='192.168.43.127', debug=True,threaded=False)
