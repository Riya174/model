import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	#USING TO DISABLE TENSORFLOW MESSAGES
import tensorflow as tf
import cv2
import imutils
from operator import itemgetter

def TextboxDetector(imgOrig):
	with tf.gfile.Open('frozen_inference_graph_Textboxes.pb', 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	with tf.Session() as sess:
		sess.graph.as_default()
		tf.import_graph_def(graph_def, name='')
		List = []
		for angle in np.arange(0, 180, 3):
			img = imutils.rotate_bound(imgOrig, angle)
			
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

			# Visualize detected bounding boxes.
			num_detections = int(out[0][0])
					
			for i in range(num_detections):
				if i < 1:
					classId = int(out[3][0][i])
					# print(classId)
					score = float(out[1][0][i])
					bbox = [float(v) for v in out[2][0][i]]
					x = bbox[1] * cols
					y = bbox[0] * rows
					right = bbox[3] * cols
					bottom = bbox[2] * rows
					widthNew, heightNew, channels = img[int(y): int(bottom), int(x): int(right)].shape
					widthNew = float(widthNew)
					heightNew = float(heightNew)
					List.append([widthNew/heightNew,img[int(y): int(bottom), int(x): int(right)]])
			if 	widthNew/heightNew < 0.25:
				break
	sess.close()
	tf.reset_default_graph()
	textbox_img = min(List,key=itemgetter(0))[1]
	return textbox_img

def Augmenting(textbox_rect):
	avg_per_row = np.average(textbox_rect, axis = 0)
	avg_color = np.average(avg_per_row, axis = 0)
	shape = textbox_rect.shape
	if shape[0] > shape[1]:
		new_dim = shape[0]
	else:
		new_dim = shape[1]
	new_image = np.zeros((new_dim, new_dim, 3), np.uint8)
	new_image[:] = avg_color
	for x in range(shape[0]):
		for y in range(shape[1]):
			new_image[int(new_dim/2-shape[0]) + x][y] = textbox_rect[x][y]
	return new_image

def ImageImprov(augmented, gamma=2, alpha=1.5, doSmoothing=1, kernel_size=20, smoothing_param=100):
	lookUpTable = np.empty((1,256), np.uint8)
	for i in range(256):
		lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
	gamma_corr = cv2.LUT(augmented, lookUpTable)
	contrasted = cv2.convertScaleAbs(gamma_corr, alpha = alpha)
	if doSmoothing == 1:	
		smooth = cv2.bilateralFilter(contrasted, kernel_size, smoothing_param, smoothing_param)
	else:
		smooth = contrasted.copy()
	return smooth

