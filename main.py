import numpy as np
import tensorflow as tf
import cv2
import time
from imutils.video import FPS

# Importing the helper functions
from helpers import ops as helper_ops
from helpers import label_map_helper
from helpers import visualization_utils as vis_util

# Setting up the font for the openCV texts on display
font = cv2.FONT_HERSHEY_DUPLEX

# Setting up configurations
helper_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
category_index = label_map_helper.create_category_index_from_labelmap('./data/mscoco_label_map.pbtxt',
                                                                      use_display_name=True)
# Data set folder name which was previously trained and download from the tensorflow datasets
modelName = 'ssdlite_mobilenet_v2_coco_2018_05_09'
modelDirectory = f"./models/{modelName}/saved_model"

# Loading the trained data model set
detectionModel = tf.saved_model.load(str(modelDirectory))
detectionModel = detectionModel.signatures['serving_default']

# Global variable to store the crash Frame count
crashFrameCount = 0


# Function to process the single image from the video
# Here the image is converted to a numpy array and then it converts to
# a tensorflow object using the numpy array.
# Then by comparing with the trained data sets it generated the number of detections the
# image with comparing the labels
def processSingleImage(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    processedDictionary = model(input_tensor)
    numDetections = int(processedDictionary.pop('num_detections'))
    processedDictionary = {key: value[0, :numDetections].numpy()
                           for key, value in processedDictionary.items()}
    processedDictionary['num_detections'] = numDetections
    processedDictionary['detection_classes'] = processedDictionary['detection_classes'].astype(np.int64)
    return processedDictionary


# Here the function is generating the output image to show on the screen.
# Inside the function it checks whether the current object is collidable with the
# vechicle and then it draws the corresponsing boxes and the scores on the objects.
def showProcessedImage(model, image_path):
    image_np = np.array(image_path)
    height, width, channel = image_np.shape
    processedDictionary = processSingleImage(model, image_np)
    predictCollision(processedDictionary, height, width, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        processedDictionary['detection_boxes'],
        processedDictionary['detection_classes'],
        processedDictionary['detection_scores'],
        category_index,
        instance_masks=processedDictionary.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


# In this function it predicts whether there will be a collision or not.
# If the object is going to collide with the vehicle it generates an output text "WARNING" to the screen.
def predictCollision(processedDictionary, height, width, image_np):
    global crashFrameCount
    is_crashed = 0
    max_area = 0
    details = [0, 0, 0, 0]
    for ind, scr in enumerate(processedDictionary['detection_classes']):
        if scr in [2, 3, 4, 6, 8]:
            ymin, xmin, ymax, xmax = processedDictionary['detection_boxes'][ind]
            score = processedDictionary['detection_scores'][ind]
            if score > 0.5:
                obj_area = int((xmax - xmin) * width * (ymax - ymin) * height)
                if obj_area > max_area:
                    max_area = obj_area
                    details = [ymin, xmin, ymax, xmax]

    x_center, y_center = (details[1] + details[3]) / 2, (details[0] + details[2]) / 2
    if max_area > 70000 and ((x_center < 0.2 and details[2] > 0.9) or
                             (0.2 <= x_center <= 0.8) or
                             (x_center > 0.8 and details[2] > 0.9)):
        is_crashed = 1
        crashFrameCount = 15

    if is_crashed == 0:
        crashFrameCount = crashFrameCount - 1

    if crashFrameCount > 0:
        cv2.putText(image_np, "WARNING!!!", (100, 100), font, 4, (255, 0, 0), 3, cv2.LINE_AA)


# OpenCV operation to get the video stream and output the video on to the screen.
cap = cv2.VideoCapture('input-video-2.mp4')
time.sleep(2.0)
cap.set(1, 0)
fps = FPS().start()
processed_count = 0
while True:
    success, image = cap.read()
    if not success:
        break
    image = showProcessedImage(detectionModel, image)
    cv2.imshow("Output Video", image)
    fps.update()
    key = cv2.waitKey(1)

fps.stop()
cap.release()
cv2.destroyAllWindows()
