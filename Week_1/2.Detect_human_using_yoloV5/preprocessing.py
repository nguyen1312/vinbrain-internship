import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import xml.etree.ElementTree as ET

def make_datapath_list(mode = "Train/"):
    rootpath = "./pedestrian-dataset/"
    target_path = osp.join(rootpath + mode + mode + "Annotations"+ "/*.xml")
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)
    return sorted(path_list)

def make_labels(mode = "Train/"):
    list_anno = make_datapath_list(mode) # return list annotation file
    for each_annotation in list_anno[0:1]:
        xml = ET.parse(each_annotation).getroot()
        ret = []
        width = 0
        height = 0
        filename = xml.find("filename").text
        print(filename)
        for obj in xml.iter("size"):
            width = int(obj.find("width").text)
            height = int(obj.find("height").text)
        for obj in xml.iter('object'):
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue    
            bndbox = []
            bndbox.append(0)
            bbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            for pt in pts:
                pixel = int(bbox.find(pt).text) - 1
                if pt == "xmin" or pt == "xmax":
                    pixel /= width # ratio of width
                else:
                    pixel /= height # ratio of height 
                bndbox.append(pixel)
                # ( class xmin ymin xmax ymax )
            x_center = (bndbox[1] + bndbox[3]) / 2
            y_center = (bndbox[2] + bndbox[4]) / 2
            width = bndbox[3] - bndbox[1]
            height = bndbox[4] - bndbox[2]
            modify_bndbox = [0, x_center, y_center, width, height]
            ret.append(" ".join(map(str, modify_bndbox)))
    
        filename = filename[:-3] + "txt"
        for new_line in ret:
            with open("./output/val/" + filename, "a") as a_file:
                a_file.write(new_line)
                a_file.write("\n")
        



if __name__ == "__main__": 
    make_labels("Val/")
