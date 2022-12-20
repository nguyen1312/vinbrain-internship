from lib import *

def detectImage(img, model):
    image = Image.open(img)
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(image)        
    return detections.xyxy[0]