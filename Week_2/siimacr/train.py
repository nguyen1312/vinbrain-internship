from lib import *
from metric import Meter
def train(epoch, device, model, train_dataloader, accumulation_steps, optimizer):
    meter = Meter("train", epoch)
    model.train(True)
    running_loss = 0.0
    optimizer.zero_grad()
    total_batches = len(train_dataloader)
    for itr, batch in enumerate(train_dataloader):
        images, targets = batch
        images = images.to(device)
        masks = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss = loss / accumulation_steps    
        loss.backward()
        if (itr + 1 ) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        outputs = outputs.detach().cpu()
        meter.update(targets, outputs)

    
    epoch_loss = (running_loss * accumulation_steps) / total_batches
    dice, iou = epoch_log("train", epoch, epoch_loss, meter)
    torch.cuda.empty_cache()
    return epoch_loss, dice, iou