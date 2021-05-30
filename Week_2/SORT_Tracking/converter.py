from lib import *

# convert bbox Yolo format (x_min, y_min, x_max, y_max, ...) into vector z kalmanFilter (x_center, y_center, scale, ratio) 
def convert_bbox_to_z(bbox):
    # bbox[:4] -> 4 first elements in bbox are x1,y1,x2,y2
    x1, y1, x2, y2 = bbox[:4]
    width = x2 - x1 
    height = y2 - y1
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    # scale: dien tich hcn
    scale = width * height
    # ratio: ty le width / height
    ratio = width / float(height)
    return np.array([x_center, y_center, scale, ratio]).reshape((4, 1))

# convert vector x kalmanFilter (x_center, y_center, scale, ratio, ...) into bbox Yolo format (x_min, y_min, x_max, y_max, ...)  
def convert_x_to_bbox(x, score = None):
    # x[:4] -> 4 first elements in bbox are x_center, y_center, scale, ratio
    x_center, y_center, scale, ratio = x[:4]
    # scale = width * height, ratio = width / height
    width = np.sqrt(scale * ratio)
    height = scale / width
    # width = x2 - x1, height = y2 - y1, x_center = (x1 + x2)/2, y_center = (y1+y2)/2
    x_min = (2 * x_center - width) / 2
    x_max = (2 * x_center + width) / 2
    y_min = (2 * y_center - height) / 2
    y_max = (2 * y_center + height) / 2
    if score is None:
        return np.array((x_min, y_min, x_max, y_max)).reshape((1, 4))
    else:
        return np.array((x_min, y_min, x_max, y_max)).reshape(1, 5)