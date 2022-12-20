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
        _, img_in = cap.read()
        if img_in is None:
            time.sleep(0.1)
            count += 1
            if count < 3:
                continue
            else: 
                break
        t1 = time.time()
        with torch.no_grad():
            detections = model(img_in)

        tensorBBox = detections.xyxy[0].detach().numpy()
        bboxPerson = tensorBBox[np.where(tensorBBox[:, -1] == 0)]
        track_bbs_ids = tracker.update(bboxPerson)
        for track_object in track_bbs_ids:  
            x1 = int(track_object[0])
            y1 = int(track_object[1])
            x2 = int(track_object[2])
            y2 = int(track_object[3])
            track_label = str(int(track_object[4])) 
            cv2.rectangle(img_in, (x1, y1), (x2, y2), (0, 255, 255), 4)
            cv2.putText(img_in, '#' + track_label, (x1 + 5, y1 - 10), 0, 3, (0, 40, 255), thickness = 3)
        # track_img = inferenceTrackFromFrame(tracker, img_in, model)
        # track_img = cv2.resize(track_img, dsize=(700, 700))
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img_in, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        # cv2.imshow('output', img_in)
        cv2.imshow("Preview", img_in)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close down the video stream
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    print(__doc__)
    main()