from lib import *

def iou(bbox1, bbox2):
    x1, y1 = np.maximum(bbox1[0:2], bbox2[0:2])
    x2, y2 = np.minimum(bbox1[2:4], bbox2[2:4])

    width = np.maximum(0, x2 - x1)
    height = np.maximum(0, y2 - y1)

    # dien tich hcn: lay dai nhan rong
    union = width * height 
    
    # dien tich hcn 2 bbox
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # dien tich hop nhau bbox1 & bbox2
    intersection = area_bbox1 + area_bbox2
    return union / intersection
