import numpy as np
import tensorflow as tf
import cv2
import time
from imutils.video import FPS
from helpers import ops as helper_ops
from helpers import label_map_helper
from helpers import visualization_utils as vis_util

font = cv2.FONT_HERSHEY_DUPLEX

helper_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
category_index = label_map_helper.create_category_index_from_labelmap('./data/mscoco_label_map.pbtxt',
                                                                      use_display_name=True)
modelName = 'ssdlite_mobilenet_v2_coco_2018_05_09'
modelDirectory = f"./models/{modelName}/saved_model"
detectionModel = tf.saved_model.load(str(modelDirectory))
detectionModel = detectionModel.signatures['serving_default']
crash_count_frames = 0


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


def predictCollision(processedDictionary, height, width, image_np):
    global crash_count_frames
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
        crash_count_frames = 15

    if is_crashed == 0:
        crash_count_frames = crash_count_frames - 1

    if crash_count_frames > 0:
        cv2.putText(image_np, "WARNING !!!", (100, 100), font, 4, (255, 0, 0), 3, cv2.LINE_AA)


cap = cv2.VideoCapture('input_dash.mp4')
time.sleep(2.0)

cap.set(1, 0)

fps = FPS().start()

processed_count = 0
while True:
    (grabbed, frame) = cap.read()
    frame = frame[:-150, :, :]
    processed_count = processed_count + 1
    if processed_count == 3334:
        break
    frame = showProcessedImage(detectionModel, frame)

    cv2.imshow("version", frame)
    fps.update()

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break

fps.stop()
cap.release()
cv2.destroyAllWindows()
