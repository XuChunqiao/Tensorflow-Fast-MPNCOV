import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import vision1.torchvision.transforms as transforms
import vision1.torchvision.datasets as datasets
import vision1.torchvision.models as models
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
from torch.autograd import Variable


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch BCNN Training')
parser.add_argument('--data', metavar='DIR', default='/media/zzm/TXQ_500G/xcq/datasets/cub',   #fgvc-aircraft-2013b \cub\cars
                    help='path to dataset')

parser.add_argument('--arch', '-a', metavar='ARCH', default='isqrt_resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: vgg_m)')
parser.add_argument('--weight-path', metavar='DIR',
                    default='/media/zzm/TXQ_500G/xcq/fast-MPN-COV-ResNet-50.pth',
                    help='path to weights of pretrained model ')
parser.add_argument('--workers', '-j', default=8, type=int, metavar='N',
                    help='number of data loading workers(default:4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epoches to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', '-b', default=16, type=int, metavar='N',
                    help='mini-batch size(default: 64)')
parser.add_argument('--learning-rate', '-lr', default=2.2e-3
                    , type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum(default:0.9)')
parser.add_argument('--weight-decay', '-wd', default=1e-4, type=float, metavar='W',
                    help='weight decay(default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',       # /media/zzm/TXQ_500G/xcq/exp/cars/isqrt_resnet50_46_0.0065/net-epoch-31.pth.tar
                    help='path to latest checkpoint (default:none)')
parser.add_argument('--evaluate', '-e', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for innitializing training')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--scale', default=2, type=int,
                    help='image scale')
parser.add_argument('--dataAugmentation', default='f1', type=str,
                    help='dataset augmentation')
parser.add_argument('--exp', metavar='DIR', default='/media/zzm/TXQ_500G/xcq/exp',
                    help='path to experiment result')
# parser.add_argument('--checkppint', metavar='DIR', default='/media/xcq/Seagate Backup Plus Drive/data/exp/bcnn_train',
#                     help='path to checkpoint of bcnn_model')

global args
args = parser.parse_args()
best_prec1 = 0



class stats:
    def __init__(self, path, start_epoch):
        if start_epoch is not 0:
           stats_ = sio.loadmat(os.path.join(path, 'stats.mat'))
           data = stats_['data']
           content = data[0, 0]
           self.trainObj = content['trainObj'][:,:start_epoch].squeeze().tolist()
           self.trainTop1 = content['trainTop1'][:,:start_epoch].squeeze().tolist()
           self.trainTop5 = content['trainTop5'][:,:start_epoch].squeeze().tolist()
           self.valObj = content['valObj'][:,:start_epoch].squeeze().tolist()
           self.valTop1 = content['valTop1'][:,:start_epoch].squeeze().tolist()
           self.valTop5 = content['valTop5'][:,:start_epoch].squeeze().tolist()
        else:
           self.trainObj = []
           self.trainTop1 = []
           self.trainTop5 = []
           self.valObj = []
           self.valTop1 = []
           self.valTop5 = []


def main():
    global best_prec1

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    # pretrained model parameters
    if args.weight_path[-3:] == 'tar':
        state_dict = torch.load(args.weight_path)['state_dict']
    elif args.weight_path[-3:] == 'pth':
        state_dict = torch.load(args.weight_path)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v  # load params model.load_state_dict(new_state_dict)
    # initialize model
    model = models.__dict__[args.arch]()
    model.load_state_dict(new_state_dict)

    # create model
    if args.data[-3:] == 'cub': classes = 200
    elif args.data[-4:] == 'cars': classes = 196
    elif args.data[-19:] == 'fgvc-aircraft-2013b':classes = 100
    model.fc = nn.Linear(int(256*(256+1)/2), classes)# initialize a classifier--isqrt_resnet
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()



    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    # get image information
    meta = {}

    if args.data[-3:] == 'cub': meta_name ='cub_meta'
    elif args.data[-4:] == 'cars': meta_name = 'cars_mata'
    elif args.data[-19:] == 'fgvc-aircraft-2013b': meta_name = 'aircraft_mata'
    if os.path.exists(os.path.join(args.exp, meta_name+'.pt')):
        meta = torch.load(os.path.join(args.exp, meta_name+'.pt'))
    else:
        average = get_image_average()
        meta['imageSize'] = [224, 224, 3]
        meta['imageAverage'] = average/255.0
        meta['std'] = [1/255.0, 1/255.0, 1/255.0]
        with open(os.path.join(args.exp, meta_name+'.pt'), 'wb') as f:
            torch.save(meta, f)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    param = [
        {'params': nn.Sequential(*list(model.children())[:11]).parameters()},
        {'params': model.fc.parameters(), 'lr':  5*args.learning_rate}
    ]

    optimizer = torch.optim.SGD(param, args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=meta['imageAverage'],
                                     std=meta['std'])
    imageSize = meta['imageSize'][0] * args.scale
    if args.data[-3:] == 'cub':

        # traindir = os.path.join(args.data, 'train')
        # valdir = os.path.join(args.data, 'val')
        #
        # train_dataset = datasets.ImageFolder(root=traindir,
        #                              transform=transforms.Compose([
        #                                  transforms.Resize(imageSize),
        #                                  transforms.CenterCrop(imageSize),
        #                                  transforms.RandomHorizontalFlip(),
        #                                  transforms.ToTensor(),
        #                                  normalize]))
        # test_dataset = datasets.ImageFolder(root=valdir,
        #                             transform=transforms.Compose([
        #                                 transforms.Resize(imageSize),
        #                                 transforms.CenterCrop(imageSize),
        #                                 transforms.ToTensor(),
        #                                 normalize]))

        train_dataset = datasets.CUB(root=args.data,
                                     transform=transforms.Compose([
                                         transforms.Resize(imageSize),
                                         transforms.CenterCrop(imageSize),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize]),
                                     train=1)
        test_dataset = datasets.CUB(root=args.data,
                                    transform=transforms.Compose([
                                        transforms.Resize(imageSize),
                                        transforms.CenterCrop(imageSize),
                                        transforms.ToTensor(),
                                        normalize]),
                                    train=0)
    elif args.data[-4:] == 'cars':
        train_dataset = datasets.CARS(root=args.data,
                                      transform=transforms.Compose([
                                         transforms.Resize(imageSize),
                                         transforms.CenterCrop(imageSize),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize]),
                                      train=1)
        test_dataset = datasets.CARS(root=args.data,
                                     transform=transforms.Compose([
                                        transforms.Resize(imageSize),
                                        transforms.CenterCrop(imageSize),
                                        transforms.ToTensor(),
                                        normalize]),
                                     train=0)
    elif args.data[-19:] == 'fgvc-aircraft-2013b':
        train_dataset = datasets.Aircrafts(root=args.data,
                                           transform=transforms.Compose([
                                               transforms.Resize((512, 512)),
                                               transforms.CenterCrop(imageSize),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize]),
                                           train=1)
        test_dataset = datasets.Aircrafts(root=args.data,
                                          transform=transforms.Compose([
                                              transforms.Resize((512, 512)),
                                              transforms.CenterCrop(imageSize),
                                              transforms.ToTensor(),
                                              normalize]),
                                          train=0)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    # make director for store checkpoint files
    if args.data[-3:] == 'cub':
        resnet_train_dir = os.path.join(args.exp, 'cub', args.arch+'_'+str(args.batch_size)+'_'+str(args.learning_rate))
    elif args.data[-4:] == 'cars':
        resnet_train_dir = os.path.join(args.exp, 'cars', args.arch+'_'+str(args.batch_size)+'_'+str(args.learning_rate))
    elif args.data[-19:] == 'fgvc-aircraft-2013b':
        resnet_train_dir = os.path.join(args.exp, 'aircraft',args.arch + '_' + str(args.batch_size) + '_' + str(args.learning_rate))
    if not os.path.exists(resnet_train_dir):
        os.makedirs(resnet_train_dir)

    stats_ = stats(resnet_train_dir, args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        # valObj, prec1, prec5 = validate(val_loader, bcnn_model, criterion)
        trainObj1, top1, top5 = train(train_loader, model, criterion, optimizer, epoch)
        stats_.trainObj.append(trainObj1)
        stats_.trainTop1.append(top1.cpu().numpy())
        stats_.trainTop5.append(top5.cpu().numpy())
        # evaluate on validation set
        valObj1, prec1, prec5 = validate(val_loader, model, criterion)
        stats_.valObj.append(valObj1)
        stats_.valTop1.append(prec1.cpu().numpy())
        stats_.valTop5.append(prec5.cpu().numpy())
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        filename = os.path.join(resnet_train_dir, 'net-epoch-%s.pth.tar' % (epoch + 1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename)
        plot_cruve(stats_, resnet_train_dir, True)
        data = stats_
        sio.savemat(os.path.join(resnet_train_dir, 'stats.mat'), {'data': data})

    model.cpu()

    # # SVM clasifier
    # result_SVM_ft = os.path.join(args.exp, 'result_svm_ft')
    # SVM_train_test(model, meta, result_SVM_ft)

def plot_cruve(stats, path, iserr):
    trainObj = np.array(stats.trainObj)
    valObj = np.array(stats.valObj)
    if iserr:
        trainTop1 = 100 - np.array(stats.trainTop1)
        trainTop5 = 100 - np.array(stats.trainTop5)
        valTop1 = 100 - np.array(stats.valTop1)
        valTop5 = 100 - np.array(stats.valTop5)
        titleName = 'error'
    else:
        trainTop1 = np.array(stats.trainTop1)
        trainTop5 = np.array(stats.trainTop5)
        valTop1 = np.array(stats.valTop1)
        valTop5 = np.array(stats.valTop5)
        titleName = 'accuracy'
    epoch = len(trainObj)
    figure = plt.figure()
    obj = plt.subplot(1, 3, 1)
    obj.plot(range(1, epoch + 1), trainObj, 'o-', label='train')
    obj.plot(range(1, epoch + 1), valObj, 'o-', label='val')
    plt.xlabel('epoch')
    plt.title('objective')
    handles, labels = obj.get_legend_handles_labels()
    obj.legend(handles[::-1], labels[::-1])
    top1 = plt.subplot(1, 3, 2)
    top1.plot(range(1, epoch + 1), trainTop1, 'o-', label='train')
    top1.plot(range(1, epoch + 1), valTop1, 'o-', label='val')
    plt.title('top1' + titleName)
    plt.xlabel('epoch')
    handles, labels = top1.get_legend_handles_labels()
    top1.legend(handles[::-1], labels[::-1])
    top5 = plt.subplot(1, 3, 3)
    top5.plot(range(1, epoch + 1), trainTop5, 'o-', label='train')
    top5.plot(range(1, epoch + 1), valTop5, 'o-', label='val')
    plt.title('top5' + titleName)
    plt.xlabel('epoch')
    handles, labels = top5.get_legend_handles_labels()
    top5.legend(handles[::-1], labels[::-1])
    filename = os.path.join(path, 'net-epoch.pdf')
    figure.savefig(filename, bbox_inches='tight')






def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        epoch, i, len(train_loader), batch_time=batch_time,
        data_time=data_time, loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def get_image_average():
    trans = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(256)])
    if args.data[-3:] == 'cub':
        dataset = datasets.CUB(root=args.data, transform=trans, train=1)
    elif args.data[-4:] == 'cars':
        dataset = datasets.CARS(root=args.data, transform=trans, train=1)
    elif args.data[-19:] == 'fgvc-aircraft-2013b':
        dataset = datasets.Aircrafts(root=args.data, transform=trans, train=1)
    n = dataset.targets.__len__()
    rgbAverage =[]

    for i in range(n):
        if i % 128 == 0:
            print('collecting image stats: batch starting with image %d .......'%i)
        # if i % 5 == 0:
        img_path, _ = dataset.imgs[i]
        img = Image.open(img_path)
        img = np.array(trans(img))
        if len(img.shape) == 2:
            img = torch.from_numpy(img).view(256, 256, 1).expand(256, 256, 3).numpy()
        elif img.shape[2] == 1:
            img = torch.from_numpy(img).expand(256, 256, 3).numpy()
        rgbAverage.append(img.mean(axis=(0, 1)))
    rgbAverage = np.mean(np.asarray(rgbAverage), axis=0)
    # mean = torch.from_numpy(rgbAverage).view(1, 1, 3).expand(448, 448, 3).numpy()
    # cov = []
    # for i in range(n):
    #     if i % 64 == 0:
    #         print('collecting image stats: batch starting with image %d .......'%i)
    #     img_path, _ = dataset.imgs[i]
    #     img = Image.open(img_path)
    #     img = np.array(trans(img))
    #     if len(img.shape) == 2:
    #         img = torch.from_numpy(img).view(448, 448, 1).expand(448, 448, 3).numpy()
    #     elif img.shape[2] == 1:
    #         img = torch.from_numpy(img).expand(448, 448, 3).numpy()
    #     cov.append(np.mean((img-mean)**2, axis=(0, 1)))
    # cov = np.mean(np.asarray(cov), axis=0)
    # std = np.sqrt(cov)

    return rgbAverage




if __name__ == '__main__':
    main()
