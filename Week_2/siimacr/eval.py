from metric import Meter
from lib import *
def eval(epoch, device, model, val_dataloader, accumulation_steps, optimizer):
    meter = Meter("val", epoch)
    model.train(False)
    running_loss = 0.0
    total_batches = len(val_dataloader)
    optimizer.zero_grad()
    for itr, batch in enumerate(val_dataloader):
        images, targets = batch
        images = images.to(device)
        masks = targets.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss = loss / accumulation_steps    
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)

    epoch_loss = (running_loss * accumulation_steps) / total_batches
    dice, iou = epoch_log("val", epoch, epoch_loss, meter)
    torch.cuda.empty_cache()
    return epoch_loss, dice, iou