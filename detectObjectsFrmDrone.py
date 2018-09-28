import numpy as np
import os
import io
import traceback
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from subprocess import Popen, PIPE


from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO

import matplotlib
import PyQt5
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2.cv2 as cv2

from PIL import Image

from hdfs import InsecureClient
from pyspark import SparkContext, SparkConf

# This is needed since the notebook is stored in the object_detection folder.
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
MODEL_NAME = 'faster_rcnn_resnet101_lowproposals_coco_2018_01_28'
#MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28'
#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28'

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

"""opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())"""


detection_graph = tf.Graph()
with tf.device('/gpu:0'):
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)




process = Popen('hdfs dfs -ls /flume/eventdata',shell=True,stdout=PIPE, stderr=PIPE)
std_out, std_err = process.communicate()
stdstr = std_out.decode("utf-8") 
filelst = [line.rsplit(None,1)[-1] for line in stdstr.split('\n') if len(line.rsplit(None,1)) ] [1:]

sc = SparkContext("local","App")
#byteFileAsRdd = sc.binaryFiles('hdfs://localhost:9000/flume/eventdata').take(1)

#client = InsecureClient('http://localhost:50070', user='Nishith')
#def readimagehdfs(path):
#    with client.read(filelst[3]) as f:
#        return bytearray(f.read())


def readimage(path):
    with open(path, "rb") as f:
        return bytearray(f.read())

execution_path = os.getcwd()
directory = 'hdfscontentFromFlume'
#directory = '/flume/eventdata'
IMAGE_SIZE = (10,7.5)

with tf.device('/gpu:0'):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for filename in filelst:
                print('Fetching image byte stream '+filename)
                try:
                    byteFileAsRdd = sc.binaryFiles('hdfs://localhost:9000'+filename).take(1)
                    img_str = bytearray(byteFileAsRdd[0][1])
                    #img_str = readimagehdfs(filename)
                    arr = np.asarray(img_str, dtype=np.uint8)
                    image = cv2.imdecode( arr, -1)

                    #img_str = readimage(directory+'/'+filename)
                    #arr = np.asarray(img_str, dtype=np.uint8)
                    #image = cv2.imdecode( arr, -1)

                    if (type(image) is np.ndarray):
                        # Fetched image now detect objects
                        image_np = image
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                        # Each box represents a part of the image where a particular object was detected.
                        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                        # Each score represent how level of confidence for each of the objects.
                        # Score is shown on the result image, together with the class label.
                        scores = detection_graph.get_tensor_by_name('detection_scores:0')
                        classes = detection_graph.get_tensor_by_name('detection_classes:0')
                        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                        # Actual detection.
                        (boxes, scores, classes, num_detections) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})
                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=8)
 
                        cv2.imshow('object detection', image_np)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            sc.stop()
                            break

                except Exception as ex:
                    sc.stop()
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback)
                    print('Could not process '+directory+'/'+filename)
sc.stop()