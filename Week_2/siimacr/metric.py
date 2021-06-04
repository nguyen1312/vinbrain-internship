from lib import *
from predict import predict

def compute_ious(pred, label):
    ious = []
    label_c = label == 1
    if np.sum(label_c) == 0:
        ious.append(np.nan)
        
    pred_c = pred == 1
    intersection = np.logical_and(pred_c, label_c).sum()
    union = np.logical_or(pred_c, label_c).sum()
    if union != 0:
        ious.append(intersection / union)
    return ious if ious else []

def compute_iou_batch(outputs, labels):
    ious = []
    preds = np.copy(outputs) 
    labels = np.array(labels)
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label)))
    iou = np.nanmean(ious)
    return iou

def metric(probability, truth):
    batch_size = len(truth)
    with torch.no_grad():

        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)

        p = (probability > 0.5).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)

        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1)/((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

    return dice, dice_neg, dice_pos

class Meter:
    def __init__(self, phase, epoch):
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos = metric(probs, targets)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = predict(probs)
        iou = compute_iou_batch(preds, targets)
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

def epoch_log(epoch_loss, meter):
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f" % (epoch_loss, dice, dice_neg, dice_pos, iou))
    return dice, iou

# Test
if __name__ == "__main__":
    meter_epoch_2 = Meter("train", 2)
    a = torch.rand(5, 1, 512, 512)
    b = torch.rand(5, 1, 512, 512)
    meter_epoch_2.update(a, b)
    print(meter_epoch_2.get_metrics())
