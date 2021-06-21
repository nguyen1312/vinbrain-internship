from deep_sort import nn_matching
from deep_sort.tracker import Tracker 
from deep_sort.detection import Detection
import numpy as np
import torch
import torchvision.transforms as transforms
import os.path as osp
from vggface_model import *
class DeepSort_Facenet(object):
    def __init__(self, wt_path = None):	
        # self.encoder = torch.load('./ckpts/model640.pt', map_location=torch.device('cpu'))
        # self.encoder = self.encoder.cuda()
        self.encoder = VGG_16().double()
        self.encoder.load_weights("ckpts/vgg_face_torch/VGG_FACE.t7")
        self.encoder = self.encoder.eval()
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", .5, 100)
        self.tracker= Tracker(self.metric)
        self.transforms = transforms.Compose([ 
                transforms.ToPILImage(),
                transforms.Resize((128,128)),
                transforms.ToTensor()
            ])

    def reset_tracker(self):
        self.tracker= Tracker(self.metric)
    
    # def format_yolo_output(self, out_boxes):
    #     for b in range(len(out_boxes)):
    #         out_boxes[b][0] = out_boxes[b][0] - out_boxes[b][2]/2
    #         out_boxes[b][1] = out_boxes[b][1] - out_boxes[b][3]/2
    #     return out_boxes				
    # def pre_process(self,frame, detections):	
    #     crops = []
    #     for d in detections:
    #         d = d.detach().numpy().tolist()
    #         print(d)
    #         for i in range(len(d)):
    #             if d[i] <0:
    #                 d[i] = 0	

    #         img_h,img_w,img_ch = frame.shape

    #         xmin,ymin,w,h = d

    #         if xmin > img_w:
    #             xmin = img_w

    #         if ymin > img_h:
    #             ymin = img_h

    #         xmax = xmin + w
    #         ymax = ymin + h

    #         ymin = abs(int(ymin))
    #         ymax = abs(int(ymax))
    #         xmin = abs(int(xmin))
    #         xmax = abs(int(xmax))

    #         try:
    #             crop = frame[ymin:ymax,xmin:xmax,:]
    #             crop = self.transforms(crop)
    #             crops.append(crop)
    #         except:
    #             continue
    #     if crops == []:
    #         return None
    #     crops = torch.stack(crops)
    #     return crops

    def pre_process(self, frame, detections):
        crops = []
        for d in detections:
            d = d.detach().numpy().tolist()
            for i in range(len(d)):
                if d[i] <0:
                    d[i] = 0	

            img_h, img_w, img_ch = frame.shape
            xmin, ymin, xmax, ymax = d

            if xmin > img_w:
                xmin = img_w

            if ymin > img_h:
                ymin = img_h
            
            ymin = abs(int(ymin))
            ymax = abs(int(ymax))
            xmin = abs(int(xmin))
            xmax = abs(int(xmax))

            try:
                crop = frame[ymin : ymax, xmin : xmax, :]
                crop = self.transforms(crop)
                crop = torch.Tensor(crop).permute(2, 0, 1).view(1, 3, 224, 224).double()
                crop -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)
                crops.append(crop)
            except:
                continue

        if crops == []:
            return None
        crops = torch.stack(crops)
        return crops

    def run_deep_sort(self, frame, out_scores, out_boxes):
        if out_boxes == []:			
            self.tracker.predict()
            print('No detections')
            trackers = self.tracker.tracks
            return trackers

        detections = torch.Tensor(out_boxes)
        processed_crops = self.pre_process(frame, detections)
        if processed_crops is None:
            return None, None
        features = self.encoder(processed_crops)
        features = torch.squeeze(features)
        features = features.detach().cpu().numpy()

        if len(features.shape)==1:
            features = np.expand_dims(features,0)

        dets =  [Detection(bbox, score, feature)\
                    for bbox,score, feature in\
                zip(detections,out_scores, features)]

        outboxes = np.array([d.tlwh for d in dets])
        outscores = np.array([d.confidence for d in dets])
    
        self.tracker.predict()
        self.tracker.update(dets)	
        return self.tracker,dets


