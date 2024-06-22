'''
    main process for a Lottery Tickets experiments
'''
import os
import pdb
import time
import pickle
import random
import shutil
import argparse
import numpy as np  
import matplotlib.pyplot as plt
from os.path import join

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
# import torch.nn.functional as F
# import torchvision.models as models
# import torch.backends.cudnn as cudnn
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data.sampler import SubsetRandomSampler

from copy import deepcopy
from collections import defaultdict
from setproctitle import setproctitle
from email_alert import *
from scheduler import CyclicCosineDecayLR

from utils import *
from pruner import *

parser = argparse.ArgumentParser(description='PyTorch Lottery Tickets Experiments')

##################################### Dataset #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--input_size', type=int, default=32, help='size of input images')

##################################### Architecture ############################################
parser.add_argument('--arch', type=str, default='resnet18', help='model architecture')
parser.add_argument('--imagenet_arch', action="store_true", help="architecture for imagenet size samples")

##################################### General setting ############################################
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='saved_models/sparseToggle/', type=str)
parser.add_argument('--name', help='The directory used to save the trained models', default='test', type=str)

##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=150, type=int, help='number of total epochs to run')
# parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
# parser.add_argument('--decreasing_lr', default='40', help='decreasing strategy')
parser.add_argument('--init_decay_epoch', default=110, type=int, help='lr scheduler')
parser.add_argument('--min_decay_factor', default=0.01, type=float, help='lr scheduler')
parser.add_argument('--restart_interval', default=20, type=int, help='lr scheduler')
parser.add_argument('--restart_factor', default=0.1, type=float, help='lr scheduler')
parser.add_argument('--distill_weight', default=0.5, type=float, help='lr scheduler')
parser.add_argument('--sparse_cotrain', default='cocktail', type=str, help='sparse cotraining method:[cocktail,ast,us-net,ac/dc]')

##################################### Pruning setting #################################################
parser.add_argument('--pruning_times', default=10, type=int, help='overall times of pruning')
parser.add_argument('--element_rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--channel_rate', default=0.1, type=float, help='pruning rate')
parser.add_argument('--rewind_lr', default=1, type=int,  help='decreasing strategy')
parser.add_argument('--rewind_weight', default=1, type=int, help='decreasing strategy')
parser.add_argument('--rewind_epoch', default=7, type=int, help='decreasing strategy')


best_sa = 0


def optim_init(model, args):
    return torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

def sche_init(optimizer,args):
    return CyclicCosineDecayLR(optimizer,
                                    init_decay_epochs=args.init_decay_epoch,
                                    min_decay_lr=args.min_decay_factor * args.lr,
                                    restart_interval=args.restart_interval,
                                    restart_lr=args.restart_factor * args.lr
                                    )


def main():
    global args, best_sa
    args = parser.parse_args()
    print(args)
    setproctitle(args.name)
    torch.cuda.set_device(int(args.gpu))

    args.save_dir=join(os.path.expanduser('~'),args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir=join(args.save_dir,args.name)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset
    if args.sparse_cotrain=='cocktail':
        model, train_loader, val_loader, test_loader, holdout_loader = setup_model_dataset(args,holdout=0.05)
    else:
        model, train_loader, val_loader, test_loader, _ = setup_model_dataset(args)
    masked_model=MultiMaskWrapper(model)
    masked_model.cuda()

    criterion = nn.CrossEntropyLoss()
    global criterion_kd
    criterion_kd = nn.CrossEntropyLoss()

    # decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    # if args.prune_type == 'lt':
    #     print('lottery tickets setting (rewind to the same random init)')
    #     initalization = deepcopy(model.state_dict())
    # elif args.prune_type == 'pt':
    #     print('lottery tickets from best dense weight')
    #     initalization = None
    # elif args.prune_type == 'rewind_lt':
    #     print('lottery tickets with early weight rewinding')
    #     initalization = None
    # else:
    #     raise ValueError('unknown prune_type')



    optimizer=optim_init(masked_model,args)
    scheduler=sche_init(optimizer,args)





    # if args.resume:
    #     print('resume from checkpoint {}'.format(args.checkpoint))
    #     checkpoint = torch.load(args.checkpoint, map_location = torch.device('cuda:'+str(args.gpu)))
    #     best_sa = checkpoint['best_sa']
    #     start_epoch = checkpoint['epoch']
    #     all_result = checkpoint['result']
    #     start_state = checkpoint['state']
    #
    #     if start_state>0:
    #         current_mask = extract_mask(checkpoint['state_dict'])
    #         prune_model_custom(model, current_mask)
    #         check_sparsity(model)
    #         optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                                     momentum=args.momentum,
    #                                     weight_decay=args.weight_decay)
    #         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    #
    #     model.load_state_dict(checkpoint['state_dict'])
    #     # adding an extra forward process to enable the masks
    #     x_rand = torch.rand(1,3,args.input_size, args.input_size).cuda()
    #     with torch.no_grad:
    #         model(x_rand)
    #
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    #     initalization = checkpoint['init_weight']
    #     print('loading state:', start_state)
    #     print('loading from epoch: ',start_epoch, 'best_sa=', best_sa)
    #else:
    if 1:
        all_result = {}
        all_result['train_ta'] = []
        all_result['test_ta'] = []
        all_result['val_ta'] = []

        start_epoch = 0
        start_state = 0



    print('######################################## Start Standard Training Iterative Pruning ########################################')
    global initialization
    interp_dict=defaultdict(dict)
    for stage in range(start_state, args.pruning_times+1):

        print('******************************************')
        print('pruning stage', stage)
        print('******************************************')
        
        check_sparsity(masked_model)
        for epoch in range(start_epoch, args.epochs):

            print(optimizer.state_dict()['param_groups'][0]['lr'])
            acc = train(train_loader, masked_model, criterion, optimizer, epoch,stage=stage)

            if stage == 0 and epoch == args.rewind_epoch:
                torch.save(masked_model.state_dict(), os.path.join(args.save_dir, 'epoch_{}_rewind_weight.pt'.format(epoch+1)))
                initialization = deepcopy(masked_model.state_dict())
                #remove masks, leave parameters
                for key in list(initialization.keys()):
                    if 'mask' in key:
                        del initialization[key]


            # evaluate on validation set
            tacc,_ = validate(val_loader, masked_model, criterion)
            # evaluate on test set
            test_tacc,_ = validate(test_loader, masked_model, criterion)

            scheduler.step()

            all_result['train_ta'].append(acc)
            all_result['val_ta'].append(tacc)
            all_result['test_ta'].append(test_tacc)

            # remember best prec@1 and save checkpoint
            is_best_sa = tacc > best_sa
            best_sa = max(tacc, best_sa)

            save_checkpoint({
                'stage': stage,
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': masked_model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_SA_best=is_best_sa, pruning=stage, save_path=args.save_dir)

            # plot training curve
            plt.plot(all_result['train_ta'], label='train_acc')
            plt.plot(all_result['val_ta'], label='val_acc')
            plt.plot(all_result['test_ta'], label='test_acc')
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, str(stage)+'net_train.png'))
            plt.close()


        val_pick_best_epoch = np.argmax(np.array(all_result['val_ta']))
        print('* best SA = {}, Epoch = {}'.format(all_result['test_ta'][val_pick_best_epoch], val_pick_best_epoch+1))

        all_result = {}
        all_result['train_ta'] = []
        all_result['test_ta'] = []
        all_result['val_ta'] = []
        best_sa = 0
        start_epoch = 0

        element_sparsity,channel_sparsity,nm_sparsity = check_sparsity(masked_model,True)


        # store interpolation params
        if stage>0:
            interp_dict[stage]['param']=deepcopy(masked_model.param_dict())
            interp_dict[stage]['element_mask']=deepcopy(masked_model.mask_dict('element'))
            interp_dict[stage]['channel_mask']=deepcopy(masked_model.mask_dict('channel'))
            interp_dict[stage]['nm_mask']=deepcopy(masked_model.mask_dict('nm'))
            interp_dict[stage][]


        #pruning
        if stage<args.pruning_times:
            print('UMG pruning')
            if stage==1:
                NM_rate="1:2"
            elif stage==4:
                NM_rate="2:4"
            elif stage==7:
                NM_rate="4:8"
            else:
                NM_rate=None
            masked_model.UMG_prune(args.element_rate, args.channel_rate, NM_rate)

            element_sparsity,channel_sparsity,nm_sparsity = check_sparsity(masked_model,True)


            if args.rewind_weight:
                optimizer=optim_init(masked_model,args)
                scheduler=sche_init(optimizer,args)
                assert initialization is not None
                masked_model.load_state_dict(initialization,False)
            elif args.rewind_lr:
                scheduler=sche_init(optimizer,args)

    #greedy network interpolation
    best_p_list=[]
    if args.sparse_cotrain=='cocktail':
        coeff_pool=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
        cur_param_dict=interp_dict[1]['param']
        for stage in range(2,args.pruning_times+1):
            best_p=None
            best_tot_acc=0
            for p in coeff_pool:
                interp_param_dict={}
                for key in cur_param_dict:
                    interp_param_dict[key]=p*cur_param_dict[key]+(1-p)*interp_dict[stage]['param'][key]
                masked_model.model.load_state_dict(interp_param_dict,False)
                tot_acc=0
                for s in range(1,stage+1):
                    masked_model.model.load_state_dict(interp_dict[s]['element_mask'], False)
                    masked_model.model.load_state_dict(interp_dict[s]['channel_mask'], False)
                    masked_model.model.load_state_dict(interp_dict[s]['nm_mask'], False)
                    check_sparsity(masked_model,True)
                    avg_acc,_=validate(holdout_loader,masked_model,criterion)
                    tot_acc+=avg_acc
                if best_tot_acc<tot_acc:
                    best_p=p
                    best_tot_acc=tot_acc
            best_p_list.append(best_p)
            for key in cur_param_dict:
                cur_param_dict[key] = best_p * cur_param_dict[key] + (1 - best_p) * interp_dict[stage]['param'][key]
        masked_model.model.load_state_dict(cur_param_dict, False)

        #evaluate
        print('=================================================================================')
        print('start evaluating Sparse Cocktail......')
        print('element\tchannel\tnm\t')
        for stage in range(1,args.pruning_times+1):
            masked_model.model.load_state_dict(interp_dict[stage]['element_mask'], False)
            masked_model.model.load_state_dict(interp_dict[stage]['channel_mask'], False)
            masked_model.model.load_state_dict(interp_dict[stage]['nm_mask'], False)
            check_sparsity(masked_model, True)
            avg_acc,(dense_acc,element_acc,channel_acc,nm_acc)=validate(test_loader,masked_model,criterion,print_mode='mediate')

        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print("interpolation factors:")
        for i,x in enumerate(best_p_list):
            print(i,'\t',x)


def train(train_loader, masked_model, criterion, optimizer, epoch, stage=1):
    
    dense_losses = AverageMeter()
    element_losses = AverageMeter()
    channel_losses = AverageMeter()
    nm_losses = AverageMeter()


    dense_top1 = AverageMeter()
    element_top1 = AverageMeter()
    channel_top1 = AverageMeter()
    nm_top1 = AverageMeter()

    element_sparsity,channel_sparsity,nm_sparsity=check_sparsity(masked_model)

    # switch to train mode
    masked_model.train()

    for i, (image, target) in enumerate(train_loader):

        # if epoch < args.warmup:
        #     warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader))

        image = image.cuda()
        target = target.cuda()

        def train_iter(losses, top1, sparse_mode):
            if sparse_mode!='dense':
                #dynamic distill
                with torch.no_grad():
                    masked_model.change_mask_mode('dense')
                    output_dense=masked_model(image).detach()
                masked_model.change_mask_mode(sparse_mode)
                output_clean = masked_model(image)
                loss = (1-args.distill_weight)*criterion(output_clean, target)+args.distill_weight*criterion_kd(output_clean, output_dense.softmax(dim=1))

            else:
                masked_model.change_mask_mode('dense')
                output_clean = masked_model(image)
                loss = criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(sparse_mode.upper() + ' epoch: [{0}][{1}/{2}]\t'
                                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                            'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader) , loss=losses, top1=top1))

        if stage==0:
            train_iter(dense_losses,dense_top1,'dense')
        else:
            if i%2==0:
                train_iter(dense_losses,dense_top1,'dense')
            else:
                j=i//2
                if j%3==0:
                    train_iter(element_losses,element_top1,'element')
                elif j%3==1:
                    train_iter(channel_losses,channel_top1,'channel')
                elif j%3==2:
                    train_iter(nm_losses,nm_top1,'nm')


    top1 = (dense_top1.avg + element_top1.avg + channel_top1.avg + nm_top1.avg) / 4
    print('train_accuracy {top1:.3f} dense:{dense_top1.avg:.3f}, element:{element_top1.avg:.3f}, channel:{channel_top1.avg:.3f}, nm:{nm_top1.avg:.3f}'.format(
        top1=top1,
        dense_top1=dense_top1,
        element_top1=element_top1,
        channel_top1=channel_top1,
        nm_top1=nm_top1
    ))

    return top1

def validate(val_loader, masked_model, criterion, sparse_only=False, print_mode='full'):
    """
    Run evaluation
    """
    dense_losses = AverageMeter()
    element_losses = AverageMeter()
    channel_losses = AverageMeter()
    nm_losses = AverageMeter()


    dense_top1 = AverageMeter()
    element_top1 = AverageMeter()
    channel_top1 = AverageMeter()
    nm_top1 = AverageMeter()

    element_sparsity,channel_sparsity,nm_sparsity=check_sparsity(masked_model)
    # switch to evaluate mode
    masked_model.eval()

    for i, (image, target) in enumerate(val_loader):
        
        image = image.cuda()
        target = target.cuda()

        # compute output
        def val_iter(losses, top1, sparse_mode):
            masked_model.change_mask_mode(sparse_mode)

            with torch.no_grad():
                output = masked_model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0 and print_mode=='full':
                print(sparse_mode.upper() + ' test: [{0}/{1}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), loss=losses, top1=top1))

        if not sparse_only:
            val_iter(dense_losses,dense_top1,'dense')
        if element_sparsity>0:
            val_iter(element_losses,element_top1,'element')
        if channel_sparsity>0:
            val_iter(channel_losses,channel_top1,'channel')
        if nm_sparsity>0:
            val_iter(nm_losses,nm_top1,'nm')

    if not sparse_only:
        top1 = (dense_top1.avg + element_top1.avg + channel_top1.avg + nm_top1.avg) / 4
    else:
        top1 = (element_top1.avg + channel_top1.avg + nm_top1.avg) / 3

    print(
        'valid_accuracy {top1:.3f} dense:{dense_top1.avg:.3f}, element:{element_top1.avg:.3f}, channel:{channel_top1.avg:.3f}, nm:{nm_top1.avg:.3f}'.format(
            top1=top1,
            dense_top1=dense_top1,
            element_top1=element_top1,
            channel_top1=channel_top1,
            nm_top1=nm_top1
        ))

    return top1, (dense_top1.avg,element_top1.avg,channel_top1.avg,nm_top1.avg)

def save_checkpoint(state, is_SA_best, save_path, pruning, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, str(pruning)+filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, str(pruning)+'model_SA_best.pth.tar'))

# def warmup_lr(epoch, step, optimizer, one_epoch_step):
#
#     overall_steps = args.warmup*one_epoch_step
#     current_steps = epoch*one_epoch_step + step
#
#     lr = args.lr * current_steps/overall_steps
#     lr = min(lr, args.lr)
#
#     for p in optimizer.param_groups:
#         p['lr']=lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    print('setup random seed = {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        global args
        print(traceback.format_exc())
        send_an_error_message(program_name=args.name, error_name=repr(ex), error_detail=traceback.format_exc())

