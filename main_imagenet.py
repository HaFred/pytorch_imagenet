import argparse
import os
import random
import shutil
import time
import warnings
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import model as self_model
import logging
import numpy as np
import pandas as pd
from model.resnet import *
from model.alexnet import *


from ptflops import get_model_complexity_info
# from torchsummary import summary

# sys.path.append('/home/zhongad/PycharmProjects/pytorch_imagenet/')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',  # 'resnet18' lower letter for torch defined,
                    # 'ResNet18' upper letter for self defined model for EG,
                    # choices=model_names,
                    help='models architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=180, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
# note here if batch != 256 due to memory limits, the lr should decrease along with to avoid jump over optimum
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.7, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate models on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained models')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--checkpoint_save_path', default='.', type=str)
parser.add_argument('--folder_name', default='./eg_new', type=str)
parser.add_argument('-prune_rate', type=str, default='0')
parser.add_argument('-zero_grad_mea', type=bool, default=True)

best_acc1 = 0


def main():
    args = parser.parse_args()

    # fred:
    args.pretrained = False
    args.multiprocessing_distributed = False
    # args.gpu = 0
    args.data = '/home/zhongad/Downloads/imagenet/'
    args.resume = './eg_new/' + args.arch + '_checkpoint.pth.tar'
    args.cos_annl_lr_scheduler = True
    args.checkpoint_save_path = 'eg_new'

    # Logging
    if not os.path.exists('logging'):
        os.makedirs('logging')
    localtime = time.localtime(time.time())
    time_str = str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(
        localtime.tm_min)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename='./logging/' + args.arch + '_' + time_str +
                                 '_lr' + format(args.lr, '.0e') + '_log.txt',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.INFO)  # if as INFO will make the console at INFO level thus no additional stdout
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)s-%(levelname)s: %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logger = logging.getLogger()
    logger.addHandler(console)
    logging.info('Arguments:')
    logging.info(args.__dict__)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if not os.path.exists("./" + args.checkpoint_save_path):
        os.system("mkdir ./" + args.checkpoint_save_path)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print('fred: determine there is {} gpu for this node'.format(ngpus_per_node))
    # print('fred: determine world size {} for this node'.format(int(os.environ["WORLD_SIZE"])))
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, time_str, args)


def main_worker(gpu, ngpus_per_node, time_str, args):
    global best_acc1
    args.gpu = gpu
    prune_rate = float(args.prune_rate)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.empty_cache()
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_SOCKET_IFNAME'] = 'eno1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29595"
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create models
    if args.pretrained:
        print("=> using pre-trained models '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch == 'ResNet18':
        model = self_model.ResNet18(conv_act='relu',
                                    train_mode_conv='Sign_symmetry_magnitude_uniform',
                                    train_mode_fc='Sign_symmetry_magnitude_uniform',
                                    prune_flag='StochasticFA', prune_percent=prune_rate,
                                    angle_measurement=False)
    elif args.arch == 'AlexNet':
        model = self_model.AlexNet_SFA(train_mode_conv='Sign_symmetry_magnitude_uniform',
                                       train_mode_fc='Sign_symmetry_magnitude_uniform',
                                       prune_rate=prune_rate,
                                       angle_measurement=False)
    else:
        print("=> creating models '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map models to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.cos_annl_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 227, 227), as_strings=True, print_per_layer_stat=True,
                                                 verbose=True)
        logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # summary(models, input_size=(3, 227, 227), device='cpu')

    logging.info('Model:')
    logging.info(model)

    layers_zero_grad_df = pd.DataFrame()
    zero_grads_percentage_list = []
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        localtime = time.localtime(time.time())
        print('fred: Now is {}d-{}h-{}m-{}s'.format(localtime.tm_mday, localtime.tm_hour, localtime.tm_min,
                                                    localtime.tm_sec))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, layers_zero_grad_df, zero_grads_percentage_list,
              time_str, args)

        # step the scheduler
        if args.cos_annl_lr_scheduler:
            scheduler.step()

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,  # the epoch might not be best epoch, with the best_acc1 stored, the model_best
                # pth.tar won't be updated unless it is the real best in this trial
                'optimizer': optimizer.state_dict(),
            }, is_best, arch=args.arch, foldername=args.folder_name)


def train(train_loader, model, criterion, optimizer, epoch, layers_zero_grad_df, zero_grads_percentage_list,
          time_str, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    # Dumping csv for zero_grad of each layer
    if args.zero_grad_mea:
        curr_zero_grads, num_grads, layers_zero_grad_list = num_zero_error_grad(model.module)
        layers_zero_grad_df = layers_zero_grad_df.append(pd.Series(layers_zero_grad_list), ignore_index=True)
        # since we use trial start time here, the trial number no need to include
        logging.info("Number of zero_grads ({}/{})={:.2%}".format(curr_zero_grads, num_grads,
                                                                  curr_zero_grads / num_grads))
        layers_zero_grad_df.to_csv(
            './logging/zerograd_' + time_str + '_' + args.arch +
            '_prune_' + args.prune_rate + '_' + format(args.lr, '.0e') + '.csv')
        # print("Non zero indices is {}".format(non_zero_indices))
        grad_per = 100. * curr_zero_grads / num_grads
        zero_grads_percentage_list.append(np.around(grad_per, 2))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, arch, foldername='./mrmh_bp_resnet18/checkpoint.pth.tar'):
    filename = foldername + '/' + arch + '_checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        # shutil.copyfile(filename, './mrmh_bp_resnet18/model_best.pth.tar')
        shutil.copyfile(filename, foldername + '/' + arch + '_model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def num_zero_error_grad(model):
    """
    Return
     * accumulated zero gradients for all layers in this epoch
     * total gradients amount for all layers in this model
     * a 1D list of zero_grad in this epoch, i.e., layer_zero_grad_list for different layers zero grad.
    """
    layers_zero_grad_list = []
    if model is None:
        return 0

    accu_zeros, total, idx_layer = 0, 0, 0
    if isinstance(model, ResNet):
        for module in model.children():
            for layer in module:
                # for each layer
                zero_grad, sum_g = 0, 0
                if isinstance(layer, (StochasticGradPruneLinear, StochasticGradPruneConv2d)):  # for conv1 & fc6, comment this line to enable for noprune
                    flat_g = layer.error_grad.cpu().numpy().flatten()
                    zero_grad = np.sum(flat_g == 0)
                    accu_zeros += zero_grad
                    sum_g = len(flat_g)
                    total += sum_g
                    # non_zero_idices = np.where(flat_g != 0)

                    # zero_grad of this layer write into df
                    layers_zero_grad_list.append(zero_grad / sum_g)
                    # print('testing: this layer is {}, with the idx {}'.format(layer, idx_layer))
                elif isinstance(layer, BasicBlock):
                    flat_g = layer.conv1.error_grad.cpu().numpy().flatten() + layer.conv2.error_grad.cpu().numpy().flatten()
                    zero_grad = np.sum(flat_g == 0)
                    accu_zeros += zero_grad
                    sum_g = len(flat_g)
                    total += sum_g
                    # non_zero_idices = np.where(flat_g != 0)

                    # zero_grad of this layer write into df
                    layers_zero_grad_list.append(zero_grad / sum_g)
                    # print('testing: this layer is {}, with the idx {}'.format(layer, idx_layer))
                    if layer.shortcut:  # check if the sequential object shortcut is not empty
                        zero_grad, sum_g = 0, 0
                        flat_g = layer.shortcut[0].error_grad.cpu().numpy().flatten()
                        zero_grad = np.sum(flat_g == 0)
                        accu_zeros += zero_grad
                        sum_g = len(flat_g)
                        total += sum_g

                        # zero_grad of this layer write into df
                        layers_zero_grad_list.append(zero_grad / sum_g)
                        # print('testing: this layer is {}, with the idx {}'.format(layer.shortcut, idx_layer))
                        # idx_layer += 1
    elif isinstance(model, AlexNet_SFA):
        # todo it cannot be run yet, since the layers_zero_grad_list for alexnet, haven't been adapted
        for module in model.children():
            # for those modules not in the nn.sequential, for alexnetfa & alexnetsfa, wont in
            if isinstance(module, (StochasticGradPruneConv2d, StochasticGradPruneLinear)):
                flat_g = module.error_grad.cpu().numpy().flatten()
                accu_zeros += np.sum(flat_g == 0)
                total += len(flat_g)
                # non_zero_indices_list = np.where(flat_g != 0)  # this is to report where the indices are for non0 element
            elif isinstance(module, nn.Sequential):
                for layer in module:
                    # for layer in bblock:
                    if isinstance(layer, (StochasticGradPruneConv2d, StochasticGradPruneLinear)):
                        # print('yes')
                        flat_g = layer.error_grad.cpu().numpy().flatten()
                        accu_zeros += np.sum(flat_g == 0)
                        total += len(flat_g)
                        # non_zero_indices_list = np.where(flat_g != 0)
    else:
        raise ValueError('The error grad measurement supports resnet & alexnet for now')

    # print('end this epoch, the df is {}'.format(layers_zero_grad_df))

    return int(accu_zeros), int(total), layers_zero_grad_list


if __name__ == '__main__':
    main()
