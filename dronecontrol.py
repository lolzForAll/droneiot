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


# Simple take-off landing with Camera capture
def handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        print(data)



def cameraCapture(drone):

    print("cr thr: Inside camera capture")
    
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

if __name__ == '__main__':
    flyAndCapture()

