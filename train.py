##
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import ResNet

parser = argparse.ArgumentParser(description="ResNet_cifar10",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=0.01, type=float, dest='lr')
parser.add_argument("--batch_size", default=128, type=int, dest='batch_size')
parser.add_argument("--num_epochs", default=100, type=int, dest='num_epochs')
parser.add_argument("--momentum", default=0.9, type=float, dest='momentum')
parser.add_argument("--weight_decay", default=0.0001, type=float, dest='weight_decay')

parser.add_argument("--runs_dir", default="./runs", type=str, dest='runs_dir')
parser.add_argument("--checkpoint_dir", default="./checkpoint", type=str, dest='checkpoint_dir')

args = parser.parse_args()

os.makedirs(args.checkpoint_dir, exist_ok=True)

lr = args.lr
batch_size = args.batch_size
num_epochs = args.num_epochs
momentum = args.momentum
weight_decay = args.weight_decay

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transformer)
val_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())

indices = list(range(len(train_dataset)))
random.shuffle(indices)

train_indices, val_indices = indices[:45000], indices[45000:]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

Net = ResNet(n=18).to(device)

fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda x: np.argmax(x, axis=-1)
optim = torch.optim.SGD(Net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

writer_train = SummaryWriter(args.runs_dir + '/train')
writer_val = SummaryWriter(args.runs_dir + '/val')

for epoch in range(num_epochs):
    Net.train()

    train_loss_arr = []
    val_loss_arr = []
    train_acc_arr = []
    val_acc_arr = []

    for input, label in train_loader:
        input = input.to(device)
        label = label.to(device)

        output = Net(input)
        pred = fn_pred(output.detach().cpu().numpy())

        loss = fn_loss(output, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss_arr += [loss.item()]

        acc = np.mean((pred == label.cpu().numpy()).astype(np.int))
        train_acc_arr += [acc]

    Net.eval()
    for input, label in val_loader:
        input = input.to(device)
        label = label.to(device)

        output = Net(input)
        pred = fn_pred(output.detach().cpu().numpy())

        loss = fn_loss(output, label)

        val_loss_arr += [loss.item()]

        acc = np.mean((pred == label.cpu().numpy()).astype(np.int))
        val_acc_arr += [acc]

    train_loss = np.mean(np.array(train_loss_arr))
    train_acc = np.mean(np.array(train_acc_arr))
    val_loss = np.mean(np.array(val_loss_arr))
    val_acc = np.mean(np.array(val_acc_arr))

    print("EPOCH: %04d | TRAIN => loss: %.4f acc: %.4f | VAL => loss: %.4f acc: %.4f"
          % (epoch, train_loss, train_acc, val_loss, val_acc))

    writer_train.add_scalar('loss', train_loss, epoch)
    writer_val.add_scalar('loss', val_loss, epoch)
    writer_train.add_scalar('acc', train_acc, epoch)
    writer_val.add_scalar('acc', val_acc, epoch)

writer_train.close()
writer_val.close()

torch.save(Net.state_dict(), args.checkpoint_dir + "/ResNet.pth")