import argparse
parser = argparse.ArgumentParser(description='Train with multi-gpu')
parser.add_argument('-e', '--epochs', default=10, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')
parser.add_argument('--lr', default=1e-2, type=float, metavar='N')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
parser.add_argument('-n', '--n-device', default=1, type=int, metavar='N')
parser.add_argument('-d', '--dir', default='./data', type=str, metavar='STR')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(map(str, range(args.n_device)))

import sys
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms, models
from vit_pytorch import ViT
    

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dcount = torch.cuda.device_count()
    print(f'==> Device count: {dcount}')

    print('==> Preparing dataset..')
    image_size = 32
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    args.batch_size *= dcount
    
    train_ds = datasets.CIFAR10(
        args.dir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    )

    val_ds = datasets.CIFAR10(
        args.dir,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print('==> Building model..')
    model = ViT(
        image_size = image_size,
        patch_size = 4,
        num_classes = 10,
        dim = 512,
        depth = 3,
        heads = 8,
        mlp_dim = 1024
        )
    model = model.to(device)

    if dcount > 1:
        print('Parallelizing model...')
        model = nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr*dcount, momentum=0.9)

    print('==> Training model')
    for epoch in range(args.epochs):
        train(model, train_loader, epoch, optimizer, criterion, device)
        validate(model, val_loader, criterion, device)


def train(model, train_loader, epoch, optimizer, criterion, device):
    print('Epoch: %d' % (epoch+1))

    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        disp_progress('Train', i, len(train_loader), train_loss, correct, total)
    print()


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            val_loss = loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            disp_progress('Validate', i, len(val_loader), val_loss, correct, total)
    print()


def disp_progress(mode, i, n, loss, correct, total):
    i += 1
    sys.stdout.write('\r%s: %d/%d==> Loss: %.6f | Acc: %.3f%% (%d/%d)'
        % (mode, i, n, loss/i, 100.*correct/total, correct, total))


if __name__ == '__main__':
    main(args)

