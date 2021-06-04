# SIIM-ACR Pneumothorax Segmentation
- Develop a model to classify (and if present, segment) pneumothorax from a set of chest radiographic images. 
## About dataset
- There are 12047 unique images in our CSV file.

Pneumothorax

![Alt text](./img/positive.png?raw=true "Positive")

Non-Pneumothorax

![Alt text](./img/negative.png?raw=true "Negative")

The below pie chart illustrates the percentage of negative case (healthy) and positive case (unhealthy)

![Alt text](./img/chart.png?raw=true "The percentage of two cases in data")

SIIM-ACR dataset has a skewed class distribution, most of the examples belong to class Non-Pneumothorax (healthy) (approximately 78%), with only a few examples in class Pneumothorax (unhealthy). So I know it's imbalanced.

### Preprocessing 
#### Augmentations
- Import pydicom to read file .dcm from the siim.
- Import albumentations - library for image augmentation.
- Define an augmentation pipeline:
```
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
```
#### Sampled Dataset
- Due to the imbalanced dataset between non-pneumothorax and pneumothorax images, positive samples and negatives samples in each epoch are randomly re-sampled in order that the difference between the number of positives cases and negative ones is acceptable (~1000 samples).
## Metrics to evaluate semantic segmentation model
### Dice score
- 2 * the area of overlap divided by the total number of pixels in both images.
### IoU
- IoU is the area of overlap between the predicted segmentation and the ground truth divided by the area of union between the predicted segmentation and the ground truth.
## Loss
- Combining dice loss and focal loss into mixed loss, the segmentation effect can significantly improved.
### Focal loss
- The focal loss can effectively overcome the problem of segmentation of small object samples in the dataset, and its segmentation effect is better.
### Dice loss
- The dice loss only focuses on the positive sample regions and is less affected by the imbalanced distribution of the positive and negative samples.
## Model Experiment
### Unet with Resnet50 as Encoder.
- Implement UNet using pretrained Resnet50 on Imagenet as encoder. 
### Unet with SEResNext50 as Encoder.
- Using segmentation-model-pytorch library to import UNet with pretrained SEResNext50 on Imagenet as Encoder.
## Optimizer:
- Adam Optimizer
## Hyperparameters
- Num epochs: 100
- Learning rate: 1e-5
- Batch size: 4 (due to problem CUDA out of memory, I have to reduce batch size)
- alpha: 0.3 (use for loss function)
- gamma: 3.5 (use for loss function)

## Result
- I submitted my output to Kaggle successfully and here is my results:
![Alt text](./img/submit.png?raw=true "Result")
![Alt text](./img/submission.png?raw=true "Result")

## Updating
- Doing experiment on UNet-SeResnext101 encoder.
- Learning and applying RaDam optimizer to my project pipeline.
## References
1. Paper: An Improved Dice Loss for Pneumothorax Segmentation by Mining the Information of Negative Areas (2020)



