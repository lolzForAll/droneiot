from time import sleep
import tellopy
import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy
import time
import threading
import json
import requests
from PIL import Image


# Simple take-off landing with Camera capture
def handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        print(data)



def cameraCapture(drone):

    print("cr thr: Inside camera capture")
    url_flume = 'http://localhost:5140'
    headers = {'content-type': 'application/json'}
    
    try:
        container = av.open(drone.get_video_stream())
        # skip first 300 frames
        frame_skip = 300
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                cv2.imshow('Original', image)
                cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                cv2.waitKey(1)

                # Pass image to flume source 
                payload = [{'headers': {}, 'body': image }] 
                response = requests.post(url_flume, data=json.dumps(payload), headers=headers)  
                print(response.status_code, response.reason)

                frame_skip = int((time.time() - start_time)/frame.time_base)

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        cv2.destroyAllWindows()



def flyAndCapture():
    drone = tellopy.Tello()
    try:
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)

        # Connect to drone
        drone.connect()
        drone.wait_for_connection(60.0)

        # Start camera feed
        t1 = threading.Thread(target=cameraCapture, args=(drone,))
        t1.start()
        sleep(15)
        
        # Take-off
        
        drone.takeoff()
        print("motion thr: Inside motion thread, started another thread for camera stuff")
        sleep(5)
        drone.down(50)
        print("motion thr: ok")
        sleep(5)
        drone.land()
        print("motion thr: ok")
        sleep(5)
        print("motion thr: done")
    except Exception as ex:
        print(ex)
    finally:
        drone.quit()


def cameraCaptureOnly():

    print("only cr thr: Inside only camera capture")

    drone = tellopy.Tello()

    drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
    # Connect to drone
    drone.connect()
    drone.wait_for_connection(60.0)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    ImageHeight = 720
    ImageWidth = 960
    out = cv2.VideoWriter('output.avi',fourcc, 24.0, (ImageWidth,ImageHeight))

    try:
        container = av.open(drone.get_video_stream())
        frame_skip = 300
        currentFrames = 0
        maxFrames = 100

        url_flume = 'http://localhost:5140'
        headers={'Content-Type': 'application/octet-stream'}
    
        fr_time = time.time()
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
               
                # Saves for video
                out.write(image)

                # Show as video
                cv2.imshow('Video Feed', image)
                
                cv2.waitKey(1)
                

                # Pass image to flume source 
                if ((time.time() - fr_time) > 0.25):
                    msg = 'Found an image at time '+str(time.time()) 
                    success, encoded_image = cv2.imencode('.png', image)
                    
                    #payload = [{'headers': {}, 'body': msg }] 
                    #response = requests.post(url_flume, data=json.dumps(payload), headers=headers)
                    response = requests.post(url = url_flume, data=encoded_image.tobytes(), headers=headers) 
                      
                    print(response.status_code, response.reason)
                    currentFrames = currentFrames + 1
                    fr_time = time.time()

                frame_skip = int((time.time() - start_time)/frame.time_base)
                if currentFrames > maxFrames:
                    out.release()
                    cv2.destroyAllWindows()
                    break
            break
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    #flyAndCapture()
    cameraCaptureOnly()

