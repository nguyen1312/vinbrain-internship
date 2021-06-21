import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import cv2, pickle, sys
import os
import os.path as osp
import deepsort_facenet
from facenet_pytorch import MTCNN
if __name__ == '__main__':
    path_to_video = osp.relpath("./sample_video/pedestrian.mp4")
    workers = 0 if os.name == 'nt' else 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')	
    mtcnn = MTCNN(
        image_size = 640, margin = 0, min_face_size = 5,
        thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process=True,
        device = device
    )
    cap = cv2.VideoCapture(path_to_video)
    deep_sort = deepsort_facenet.DeepSort_Facenet()
    frame_id = 1
    count = 0
    while True:
        print(frame_id)		
        ret, img_in = cap.read()
        if img_in is None:
            time.sleep(0.1)
            count += 1
            if count < 3:
                continue
            else: 
                break
        if ret is False:
            frame_id += 1
            break	
        with torch.no_grad():
            detections, scores = mtcnn.detect(img_in)
        out_scores = scores.reshape(-1, 1)
        if detections is None:
            print("No dets")
            frame_id+=1
            continue

        detections = np.array(detections)
        out_scores = np.array(out_scores) 

        tracker, detections_class = deep_sort.run_deep_sort(img_in, out_scores, detections)
        if tracker is None or detections_class is None:
            continue
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlbr() #Get the corrected/predicted bounding box
            id_num = str(track.track_id) #Get the ID for the particular track.
            features = track.features #Get the feature vector corresponding to the detection.

            #Draw bbox from tracker.
            cv2.rectangle(img_in, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0, 255, 255), 4)
            cv2.putText(img_in, str(id_num),(int(bbox[0]), int(bbox[1])), 0, 3, (0, 40, 255), thickness = 3)

            #Draw bbox from detector. Just to compare.
            for det in detections_class:
                bbox = det.to_tlbr()
                cv2.rectangle(img_in,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
        img_in = cv2.resize(img_in, dsize=(700, 700))
        cv2.imshow('frame', img_in)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_id += 1