from lib import *
from transforms import ImageTransform
from rle_decode import getMaskAndImg

class SIIMDataset(Dataset):
    def __init__(self, dataframe, fnames, size, mean, std, phase = "train"):
        self.dataframe = dataframe
        self.fnames = fnames
        self.phase = phase
        self.transforms = ImageTransform(size, mean, std)

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        indice_inDataFrame = self.dataframe.index[self.dataframe['UID'] == image_id].tolist()[0]
        image, mask = getMaskAndImg(self.dataframe, indice_inDataFrame) # img, mask are arrays
        augmentedData = self.transforms(image = image, mask = mask, phase = self.phase)
        image = augmentedData['image']
        mask = augmentedData['mask']
        return image, mask 
    def __len__(self):
        return len(self.fnames)
