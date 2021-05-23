import os
import time
import argparse

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ConTNet import build_model
from optimizer import build_optimizer
from lr_scheduler import build_lr_scheduler
from criterion import build_criterion 
from data import build_loader

from utils import accuracy, reduce_tensor, resume_model, save_model
from timm.utils import AverageMeter

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='ConTNet')

    # data and model
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--arch', type=str, default='ConT-M', 
                        choices=['ConT-M', 'ConT-B', 'ConT-S', 'ConT-Ti'],
                        help='the architecture of ConTNet')

    # model hypeparameters
    parser.add_argument('--use_avgdown', type=bool, default=False, 
                        help='If True, using avgdown downsampling shortcut')
    parser.add_argument('--relative', type=bool, default=False,
                        help='If True, using relative position embedding')
    parser.add_argument('--qkv_bias', type=bool, default=True)
    parser.add_argument('--pre_norm', type=bool, default=False)

    # base setting
    parser.add_argument('--eval', default=None, type=str,
                        help='only validation')
    parser.add_argument('--batch_size', default=512, type=int, 
                        help='batch size')
    parser.add_argument('--workers', default=8, type=int, 
                        help='number of data loading workers')
    parser.add_argument('--epoch', default=200, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--warmup_epoch', default=10, type=int,
                        help='the num of epochs')
    parser.add_argument('--resume', default=None, type=str, 
                        help='resume file path')
    parser.add_argument('--init_lr', default=5e-4, type=float, 
                        help='a low initial learning rata for adamw optimizer')
    parser.add_argument('--wd', default=0.5, type=float, 
                        help='a high weight decay setting for adamw optimizer')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum for sgd')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'SGD'], 
                        help='optimizer supported by PyTorch')
    parser.add_argument('--print_freq', default=100, type=int,
                        help='frequency of printing train info')
    parser.add_argument('--save_path', default='weights', type=str,
                        help='the path to saving the checkpoints')
    parser.add_argument('--save_best', default=True, type=bool,
                        help='saveing the checkpoint has the best acc')

    # aug&reg
    parser.add_argument('--mixup', default=0.8, type=float,
                        help='using mixup and set alpha value')
    parser.add_argument('--autoaug', default='rand-m9-mstd0.5-inc1', type=str,
                        help='using auto-augmentation')
    parser.add_argument('-ls','--label-smoothing', default=0.1, type=float,
                        help='if > 0, using label-smothing')

    # distributed parallel triaining
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DDP')

    return parser.parse_args()


def launch_worker(local_rank):
    # print(local_rank)
    if not torch.cuda.is_available():
        raise ValueError(f'CPU-only training is not supported')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)  
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()

def train(loader, model, criterion, optimizer, mixup_fn, scheduler, print_freq, epoch):
    model.train()
    if dist.get_rank() == 0:
        print(f'\n=> Training epoch{epoch}')

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (images, targets) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn:
            images, targets_ = mixup_fn(images, targets)

        # forward
        outputs = model(images)

        # update acc1, acc5
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        top1.update(acc1.item(), targets.size(0))
        top5.update(acc5.item(), targets.size(0))

        # compute loss and backward
        loss = criterion(outputs, targets_)
        loss = reduce_tensor(loss)
        losses.update(loss.item(), targets_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step_update(epoch * len(loader) + i)

        # update using time
        interval = torch.tensor([time.time() - end])
        interval = reduce_tensor(interval.cuda())
        batch_time.update(interval.item())
        end = time.time()

        if i % print_freq == 0 and dist.get_rank() == 0:
            lr = optimizer.param_groups[0]['lr']
            sep = '| '
            print(f'Epoch: [{epoch}] | [{i}/{len(loader)}] lr: {lr:.8f} '+ sep +
                f'loss {losses.val:.4f} ({losses.avg:.4f}) '+ sep +
                f'Top1.acc {top1.val:6.2f} ' + sep +
                f'Top5.acc {top5.val:6.2f} ' + sep +
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f}) ' + sep
                )

@torch.no_grad()
def validate(val_loader, model, criterion, epoch=None):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    end = time.time()
    for i, (images, targets) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # forward
        outputs = model(images)

        loss = criterion(outputs, targets)
        loss = reduce_tensor(loss)
        losses.update(loss.item(), images.size(0))

        # update acc1, acc5
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        top1.update(acc1.item(), targets.size(0))
        top5.update(acc5.item(), targets.size(0))
        
        # update using time
        interval = torch.tensor([time.time() - end])
        interval = reduce_tensor(interval.cuda())
        batch_time.update(interval.item())
        end = time.time()


    if dist.get_rank() == 0:
        stat = f"epoch {epoch}" if epoch is not None else "Only"
        print(f'=> Validation {stat}')
        sep = '| '
        print(f'loss {losses.avg:.4f} '+ sep +
            f'Top1.acc {top1.avg:6.2f} ' + sep +
            f'Top5.acc {top5.avg:6.2f} ' + sep +
            f'time {batch_time.avg:.4f} ' + sep
            )

    return top1.avg, top5.avg, losses.avg

def main(config):
    # set up ddp
    launch_worker(config.local_rank)
    # build loader
    train_loader, val_loader = build_loader(config.data_path, config.autoaug, config.batch_size, config.workers)
    # build model
    model=build_model(config.arch, config.use_avgdown, config.relative, config.qkv_bias, config.pre_norm)
    model = DDP(model.cuda(), device_ids=[config.local_rank])
    # build optimizer
    optimizer=build_optimizer(model, config.optim, config.init_lr, config.wd, config.momentum)
    # build learning scheduler
    scheduler=build_lr_scheduler(config.epoch, config.warmup_epoch, optimizer, len(train_loader))
    # build criterion and mixup
    train_criterion, mixup_fn =build_criterion(config.mixup, config.label_smoothing)
    val_criterion = torch.nn.CrossEntropyLoss()
    # init acc1 and start epoch
    best_acc1 = 0.0
    start_epoch = 0

    # only validation
    if config.eval:
        if os.path.isfile(config.eval):
            model.load_state_dict(torch.load(config.eval)['model'])
            validate(val_loader, model, val_criterion)
            return 
        else:
            print(f"=> !!!!!!! no checkpoint found at '{config.eval}'\n")
            print(f"=> !!!!!!! validation is stopped")
            return

    # resume training
    if not config.resume:
        print(f"=>Training is from scratch")
    else:
        if os.path.isfile(config.resume):
            model, optimizer, scheduler, start_epoch, best_acc1 = resume_model(config.resume, model, optimizer, scheduler)
        else: 
            print(f"=> !!!!!!! no checkpoint found at '{config.resume}'\n")

    # training 
    for epoch in range(start_epoch, args.epoch):
        train_loader.sampler.set_epoch(epoch)

        train(train_loader, model, train_criterion, optimizer, mixup_fn, scheduler, config.print_freq, epoch)

        acc1, acc5, loss = validate(val_loader, model, val_criterion, epoch)

        best_acc1 = max(best_acc1, acc1)
        is_best = (best_acc1 == acc1)

        if dist.get_rank() == 0:
            print('\n******************\t',
                   f'\nBest Top1.acc {best_acc1:6.2f}\t',
                    '\n******************\t')

            # save model 
            if not config.save_best or is_best:
                save_model(config.save_path, model, optimizer, scheduler, best_acc1, epoch, is_best)

    

if __name__ == '__main__':
    # build configa
    args = parse_args()
    # launch
    main(config=args)
    print('=> Finished!')

