
if __name__ == __main__:
    losses = {phase: [] for phase in ["train", "val"]}
    iou_scores = {phase: [] for phase in ["train", "val"]}
    dice_scores = {phase: [] for phase in ["train", "val"]}
    model = 
    for epoch in range(num_epochs):
        dataloader = sampledDataset(df, 2)
        train_dataloader = dataloader["train"]
        val_dataloader = dataloader["val"]
        epoch_loss, dice, iou = train(epoch, model, train_dataloader, accumulation_steps)
        losses["train"].append(epoch_loss)
        dice_scores["train"].append(dice)
        iou_scores["train"].append(iou)
        epoch_loss, dice, iou = eval(epoch, model, val_dataloader, accumulation_steps)
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