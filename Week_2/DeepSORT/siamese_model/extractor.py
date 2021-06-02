import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from .model import Net

class Extractor(object):
    def __init__(self, checkpoint):
        self.net = Net(reid = True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(model_path, map_location = torch.device(self.device))['net_dict']
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]
                                )
        ])
    def preprocess(self, im_crops):
        im_batch = None
        for im in im_crops:
            img = cv2.resize(im.astype(np.float32)/255., self.size)
            im_batch = torch.cat([self.norm(img).unsqueeze(0)], dim = 0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self.preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

# Test
if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
