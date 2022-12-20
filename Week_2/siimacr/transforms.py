from lib import *
class ImageTransform():
    def __init__(self, size, mean, std):
      self.transform = {
          "train": A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.ShiftScaleRotate(
                          shift_limit = 0,
                          scale_limit = 0.1,
                          rotate_limit = 10,
                          p = 0.5,
                          border_mode = cv2.BORDER_CONSTANT   
                        ),
                        A.GaussNoise(),
                        A.Resize(size, size),
                        A.Normalize(mean=mean, std=std, p = 1),
                        ToTensor()
                    ]),
          "val": A.Compose([
                        A.Resize(size, size), 
                        A.Normalize(mean=mean, std=std, p = 1),
                        ToTensor()
                  ])
      }
    def __call__(self, image, mask, phase):
      return self.transform[phase](image = image, mask = mask)