from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from vggface_model import *
workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(
    image_size = 640, margin = 0, min_face_size = 5,
    thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process=True,
    device = device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
from PIL import Image
# img = Image.open("sample_image/faces.jpg")
# Get cropped and prewhitened image tensor
# img_cropped = mtcnn(img)
# # Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(img_cropped.unsqueeze(0))
# # Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))
# print(img_probs)
# print(img_probs.size())
# img_cropped = mtcnn(img)
# print(img_cropped.size())
# boxes, prob = mtcnn.detect(img) 
# print(boxes)

def extract_face(filename, required_size = (224, 224)):
    # load image from file
    pixels = plt.imread(filename)
    # create the detector, using default weights
    # detect faces in the image
    results = mtcnn.detect(pixels)
    # extract the bounding box from the first face
    x1, y1, x2, y2 = results[0][1]
    # print(x1, y1, width, height)
    # x2, y2 = int(x1 + width), int(y1 + height)
    # extract the face
    face = pixels[int(y1) : int(y2), int(x1) : int(x2)]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def get_embeddings(filename):
    # extract faces
    face = extract_face(filename)
    # convert into an array of samples
    im = torch.Tensor(face).permute(2, 0, 1).view(1, 3, 224, 224).double()
    # create a vggface model
    # yhat = resnet(samples)
    model = VGG_16().double()
    model.load_weights("ckpts/vgg_face_torch/VGG_FACE.t7")
    model.eval()
    im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)
    yhat = model(im)
    # perform prediction
    # yhat = model.predict(samples)
    return yhat
 
pixels = extract_face("sample_image/faces.jpg")
# plt.imshow(pixels)
# plt.show()
output = get_embeddings("sample_image/faces.jpg")
print(torch.squeeze(output).size())
 