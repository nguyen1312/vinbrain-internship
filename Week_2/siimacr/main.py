import yaml
from lib import *
import sys
sys.path.append('experiment/unet-resnet50')
from unet_resnet50 import UNet
from loss import MixedLoss
from dataloader import *
from train import train
from eval import eval
from metric import Meter
import argparse

losses = {phase: [] for phase in ["train", "val"]}
iou_scores = {phase: [] for phase in ["train", "val"]}
dice_scores = {phase: [] for phase in ["train", "val"]}

with open("config/hyperparameter.yaml") as file:
    hyperparams = yaml.load(file, Loader = yaml.FullLoader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose type of model')
    parser.add_argument('--type', type = int, default = 0)
    args = parser.parse_args()
    if args.type == 0:
        model = smp.Unet('se_resnext50_32x4d', encoder_weights="imagenet", activation = None)
    elif args.type == 1:
        model = UNet()
    else:
        raise ValueError 
        
    df = pd.read_csv('preprocessing_data.csv')
    num_epochs = hyperparams["NUM_EPOCH"]
    lr = hyperparams["LEARNING_RATE"]
    k_fold = hyperparams["K_FOLD"]
    device = torch.device("cpu")
    criterion = MixedLoss(0.3, 3.5)
    optimizer = optim.Adam(model.parameters(), lr=float(lr))
    accumulation_steps = 32 // 4
    best_loss = float("inf")

    for epoch in range(num_epochs):
        dataloader = sampledDataset(df, k_fold)
        train_dataloader = dataloader["train"]
        val_dataloader = dataloader["val"]
        epoch_loss, dice, iou = train(epoch, device, model, train_dataloader, accumulation_steps, optimizer)
        losses["train"].append(epoch_loss)
        dice_scores["train"].append(dice)
        iou_scores["train"].append(iou)
        epoch_loss, dice, iou = eval(epoch, device, model, val_dataloader, accumulation_steps, optimizer)
        losses["val"].append(epoch_loss)
        dice_scores["val"].append(dice)
        iou_scores["val"].append(iou)
        if epoch_loss < best_loss:
            print("-------OH YEAH---------")
            best_loss = epoch_loss
            state = {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
            }
            print("-----------------------")
            torch.save(state, "ckpts/model_v3.pth")
            print()