from lib import *
from detectionYoloV5 import detectImage
from sort import SORT

def inferenceTrackFromFrame(tracker, input_image, modelDetection):
    with torch.no_grad():
        detections = modelDetection(input_image)

    tensorBBox = detections.xyxy[0].detach().numpy()
    bboxPerson = tensorBBox[np.where(tensorBBox[:, -1] == 0)]

    track_bbs_ids = tracker.update(bboxPerson)

    for track_object in track_bbs_ids:  
          x1 = int(track_object[0])
          y1 = int(track_object[1])
          x2 = int(track_object[2])
          y2 = int(track_object[3])
          track_label = str(int(track_object[4])) 
          cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 255, 255), 4)
          cv2.putText(input_image, '#' + track_label, (x1 + 5, y1 - 10), 0, 3, (0, 40, 255), thickness = 3)
    return input_image