import argparse
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import Net

def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Train on market1501")
    parser.add_argument("--data-dir", default = 'data', type = str)
    parser.add_argument("--no-cuda", action = "store_true")
    parser.add_argument("--gpu-id", default = 0, type=int)
    parser.add_argument("--lr", default = 0.1, type=float)
    parser.add_argument("--interval", '-i', default = 20, type=int)
    parser.add_argument('--resume', '-r', action = 'store_true')
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda:{}".format(args.gpu_id)  
    else: 
        device = "cpu"
    if torch.cuda.is_available() and not args.no_cuda:
        cudnn.benchmark = True
    
    root = args.data_dir
    # train dataloader
    train_dir = os.path.join(root, "train")
    transform_train = torchvision.transforms.Compose([
        transforms.RandomCrop((128, 64), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225]
        )
    ])
    trainloader = DataLoader(
        datasets.ImageFolder(train_dir, transform = transform_train),
        batch_size=64, shuffle=True
    )

    # # test dataloader
    # test_dir = os.path.join(root, "test")
    # transform_test = transforms.Compose([
    #     transforms.Resize((128, 64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         [0.485, 0.456, 0.406], 
    #         [0.229, 0.224, 0.225]
    #     )
    # ])
    # testloader = DataLoader(
    #     datasets.ImageFolder(test_dir, transform = transform_test),
    #     batch_size = 64, shuffle = False
    # )
    
    num_classes = len(trainloader.dataset.classes)
    net = Net(num_classes = num_classes)
    if args.resume:
        assert os.path.isfile(
            "./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
        print('Loading from checkpoint/ckpt.t7')
        checkpoint = torch.load("./checkpoint/ckpt.t7")
        net_dict = checkpoint['net_dict']
        net.load_state_dict(net_dict)
        best_acc = checkpoint['acc']

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum = 0.9, weight_decay = 5e-4)
    best_acc = 0.
    num_epoch = 40
    train_loss_epoch = []
    train_err_epoch = []
    for epoch in range(num_epoch):
        print("\nEpoch : %d" % (epoch + 1))
        net.train()
        train_loss = 0.
        correct = 0
        total = 0
        interval = args.interval
        start = time.time()
        for idx, batch in enumerate(trainloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += outputs.max(dim = 1)[1].eq(labels).sum().item()
            total += labels.size(0)
        end = time.time()
        timeEpoch = end - start
        accuracy = 100.*correct / total
        train_loss = train_loss / len(trainloader)
        print("==================================================================")
        print("Time: ", timeEpoch)
        print("Loss: ", train_loss)
        print("Accuracy: ", accuracy)
        print("==================================================================")

        train_loss_epoch.append(train_loss)
        train_err_epoch.append(1. - correct / total)
        if (epoch + 1) % 20 == 0:
            lr_decay()
        if epoch == 39:
            print("Saving parameters to checkpoint/ckpt.t7")
            checkpoint = {
                'net_dict': net.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(checkpoint, './checkpoint/ckpt.t7')

        