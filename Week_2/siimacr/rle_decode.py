from lib import *

# decode rle into mask of XRays Image
def run_length_decode(rle, height = 1024, width = 1024, fill_value = 1):
    if rle == "-1": 
      return np.zeros((height, width), np.float32) # negative case
    # init mask matrix
    component = np.zeros((height, width), np.float32)
    # flatten init matrix into list
    component = component.reshape(-1)
    # processing rle
    rle = rle[1: -1] # ignore character "[", "]"
    # get all value from rle, store in tempString
    tempString = []
    for eachRLE in rle.split(','):
      s = eachRLE.replace('\'', '')
      tempString.extend([int(s) for s in s.strip().split(" ")])
    # convert into numpy array  
    rle = np.asarray(tempString)
    # convert rle into mask
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start + index
        end = start + length
        component[start: end] = fill_value # value in rle is idx of pixel 1 in matrix we want
        start = end
    # reshape into mask matrix with shape (width, height)
    component = component.reshape(width, height).T
    return component

# return mask, image from .dcm and rle
def getMaskAndImg(dataframe, idx):
    ds = pydicom.dcmread(dataframe["filepath"][idx])
    # img is 2-D matrix
    img = ds.pixel_array
    # convert rle into mask
    mask = run_length_decode(dataframe["EncodedPixels"][idx]) 
    # get matrix shape (width, height, 3)
    img = np.stack((img, ) * 3, axis=-1) 
    return img, mask