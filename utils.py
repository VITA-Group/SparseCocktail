'''
    setup model and datasets
'''




import copy 
import torch
import numpy as np 
from torchvision.transforms import Normalize

from models import *
from dataset import *


__all__ = ['setup_model_dataset']


def setup_model_dataset(args,holdout=0):

    if args.dataset == 'cifar10':
        classes = 10
        normalization = Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_set_loader, val_loader, test_loader, holdout_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers,holdout=holdout)

    elif args.dataset == 'cifar100':
        classes = 100
        normalization = Normalize(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_set_loader, val_loader, test_loader, holdout_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)
    
    else:
        raise ValueError('Dataset not supprot yet !')

    if args.imagenet_arch:
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    print(model)

    return model, train_set_loader, val_loader, test_loader,holdout_loader


