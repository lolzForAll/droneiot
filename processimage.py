import os
import io
import sys
import traceback
from PIL import Image
from array import array
from imageai.Detection import ObjectDetection
import os

def readimage(path):
    with open(path, "rb") as f:
        return bytearray(f.read())

execution_path = os.getcwd()
directory = 'hdfscontentFromFlume'

for filename in os.listdir(directory):
    print('Fetching image byte stream '+directory+'/'+filename)
    try:
        bytes = readimage(directory+'/'+filename)
        image = Image.open(io.BytesIO(bytes))
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()
        detections = detector.detectObjectsFromImage(input_type="array", input_image=image, output_image_path=os.path.join(execution_path , filename+"OBJREC.png"))
        for eachObject in detections:
            print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print('Could not process '+directory+'/'+filename)