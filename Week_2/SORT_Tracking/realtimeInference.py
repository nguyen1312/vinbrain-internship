from __future__ import print_function 
import cv2 
import numpy as np 
from lib import *
from detectionYoloV5 import detectImage
from sort import SORT
from inference import inferenceTrackFromFrame

import time
def main():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    tracker = SORT()
    cap = cv2.VideoCapture(0)
    fps = 0.0
    count = 0 
    while(True):
        _, frame = cap.read()
        if frame is None:
            time.sleep(0.1)
            count += 1
            if count < 3:
                continue
            else: 
                break
        img_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t1 = time.time()
        track_img = inferenceTrackFromFrame(tracker, img_in, model)
        track_img = cv2.resize(track_img, dsize=(700, 700))
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(track_img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', track_img)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close down the video stream
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    print(__doc__)
    main()