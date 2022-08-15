import os
import datetime
import math
import time
import numpy as np
import random
import torch
import torch.utils.data
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import dvs128_gesture, cifar10_dvs
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn
import spikeformer
import torchvision
import utils
from transforms import ToFloat
import transforms
import presets
from torchvision.transforms.functional import InterpolationMode

global epoch, checkpoint


def fix_state():
    _seed_ = 2020

    random.seed(_seed_)

    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(_seed_)


def train_one_epoch(model, criterion, optimizer, data_loader, device,
                    epochs, print_freq, args, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epochs)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        if args.dataset == 'ImageNet':
            image.unsqueeze_(1)
            image = image.repeat(1, args.T, 1, 1, 1)

        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def evaluate(model, criterion, data_loader, device, args, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device)
            target = target.to(device)

            if args.dataset == 'ImageNet':
                image.unsqueeze_(1)
                image = image.repeat(1, args.T, 1, 1, 1)

            output = model(image)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5


def load_data(args):
    # Data loading code
    print("Loading data")
    dataset_dir = args.data_path
    distributed = args.distributed
    T = args.T
    dts_cache = args.dts_cache
    dataset = args.dataset

    st = time.time()
    if args.dvsaug:
        trm = transforms.DVSTransform(5, 30, 16, 0.3)
    else:
        trm = ToFloat()

    if dataset == 'DVSGesture':
        dataset_train = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=True, data_type='frame', frames_number=T,
                                                     split_by='number', transform=trm)
        dataset_test = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=False, data_type='frame', frames_number=T,
                                                    split_by='number', transform=trm)
    elif dataset == 'DVSCIFAR':
        train_set_pth = os.path.join(dts_cache, f'dvscifar_train_{T}pt')
        test_set_pth = os.path.join(dts_cache, f'dvscifar_test_{T}.pt')
        if os.path.exists(train_set_pth) and os.path.exists(test_set_pth):
            dataset_train = torch.load(train_set_pth)
            dataset_test = torch.load(test_set_pth)
        else:
            origin_set = cifar10_dvs.CIFAR10DVS(root=dataset_dir, data_type='frame', frames_number=T,
                                                split_by='number', transform=trm)

            dataset_train, dataset_test = utils.split_to_train_test_set(0.9, origin_set, 10)
            dataset_train.transform = trm
            dataset_test.transform = trm
            if not os.path.exists(dts_cache):
                os.makedirs(dts_cache)

            utils.save_on_master(dataset_test, test_set_pth)
            utils.save_on_master(dataset_train, train_set_pth)
    elif dataset == 'ImageNet':
        train_cache = os.path.join(dts_cache, f'ImageNet_train.pt')
        val_cache = os.path.join(dts_cache, f'ImageNet_val.pt')

        transform_train = presets.ClassificationPresetTrain(
            crop_size=args.crop_size,
            interpolation=InterpolationMode(args.interpolation),
            auto_augment_policy=args.auto_augment,
            random_erase_prob=args.random_erase,
        )
        transform_test = presets.ClassificationPresetEval(
            crop_size=args.crop_size, resize_size=args.resize_size,
            interpolation=InterpolationMode(args.interpolation)
        )

        if os.path.exists(train_cache) and os.path.exists(val_cache):
            dataset_train = torch.load(train_cache)
            dataset_train.transform = transform_train
            dataset_test = torch.load(val_cache)
            dataset_test.transform = transform_test
        else:
            dataset_train = torchvision.datasets.ImageFolder(
                os.path.join(args.data_path, 'train'),
                transform_train
            )
            dataset_test = torchvision.datasets.ImageFolder(
                os.path.join(args.data_path, 'val'),
                transform_test
            )

            if not os.path.exists(dts_cache):
                os.makedirs(dts_cache)

            utils.save_on_master(dataset_train, train_cache)
            utils.save_on_master(dataset_test, val_cache)
    else:
        raise NotImplementedError('Dataset Error')

    print("Took", time.time() - st)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler


def main(args):
    global checkpoint, epoch
    if not args.random:
        fix_state()
    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.

    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)

    output_dir = os.path.join(args.output_dir, f'{args.dataset}')
    output_dir = os.path.join(output_dir, f'{args.model}_b{args.batch_size}_T{args.T}')

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    if not args.cos:
        output_dir += f'_steplr{args.lr_step_size}_{args.lr_gamma}_ls{args.ls}'
    else:
        output_dir += f'_coslr{args.epochs}_ls{args.ls}'

    if args.warmup > 0:
        output_dir += f'_warmup{args.warmup}'

    if args.adam:
        output_dir += '_adam'
    elif args.adamw:
        output_dir += '_adamw'
    else:
        output_dir += '_sgd'

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    output_dir = os.path.join(output_dir, f'lr{args.lr}')

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    device = torch.device(args.device)

    dataset_train, dataset_test, train_sampler, test_sampler = load_data(args)
    print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')

    data_loader = DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    if args.dataset == 'ImageNet':
        n_input_channels = 3
    else:
        n_input_channels = 2

    if args.model in spikeformer.__dict__:
        model = spikeformer.__dict__[args.model](n_input_channels=n_input_channels, num_classes=args.classes,
                                                 img_size=args.img, num_frame=args.T)
    else:
        raise NotImplementedError('{} not implemented'.format(args.model))

    print("Creating model")
    model.to(device)
    print(model)

    if args.distributed and args.sync_bn:
        print('converting BN')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    version = int(torch.__version__.split('+')[0].split('.')[1])

    print('Preparing loss function')
    if version >= 10:
        print("Using label smothing")
        criterion = nn.CrossEntropyLoss(label_smoothing=args.ls)
    else:
        criterion = nn.CrossEntropyLoss()

    print('Preparing optimizer')
    if args.adam:
        print("Using Adam")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    elif args.adamw:
        print("Using AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    else:
        print("Using SGD")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)

    print('Preparing scaler')
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    print('Preparing learning rate scheduler')
    if not args.cos:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup -
                                                                  args.start_epoch)

    if args.warmup > -1:
        print('Warming up')

        lr_scheduler = utils.GradualWarmupScheduler(optimizer, multiplier=1,
                                                    total_epoch=args.warmup, after_scheduler=lr_scheduler)

    model_without_ddp = model
    if args.distributed:
        print('Preparing Distributed Data Parallel')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        print('DDP finished')

    if args.resume:
        print('Resuming')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device, args, header='Test:')
        return

    if args.tb and utils.is_main_process():
        print('Preparing tensorboard logging')
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,
                                                             args.print_freq, args, scaler)
        if utils.is_main_process() and train_tb_writer is not None:
            train_tb_writer.add_scalar('Train/train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('Train/train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('Train/train_acc5', train_acc5, epoch)

        lr_scheduler.step(epoch)
        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device, args, header='Test:')
        if te_tb_writer is not None:
            if utils.is_main_process():
                te_tb_writer.add_scalar('Test/test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('Test/test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('Test/test_acc5', test_acc5, epoch)

        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True

        if output_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1, 'test_acc5_at_max_test_acc1',
              test_acc5_at_max_test_acc1)
        print(output_dir)
    if output_dir:
        utils.save_on_master(
            checkpoint,
            os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

    return max_test_acc1


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default=r'./DVSCIFAR', help='path to dataset')
    parser.add_argument('--crop-size', default=224, type=int, help='crop size')
    parser.add_argument('--resize-size', default=256, type=int, help='resize resize')

    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.25, type=float, help="random erasing probability (default: 0.25)")

    parser.add_argument('--classes', default=10, type=int, help='number of classes')
    parser.add_argument('--img', default=128, type=int, help='size of input image')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--dataset', default='DVSCIFAR', help='choose a dataset')
    parser.add_argument('--dts_cache', default='./dts_cache', help='cache path')
    parser.add_argument('--model', default='PLIF_spikeformer_7_3x2x3', help='model variant')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='training batch size')
    parser.add_argument('--epochs', default=600, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.001)
    parser.add_argument('--ls', type=float, help='label smoothing', default=0.14)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 0.0001)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=192, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--warmup', default=25, type=int, help='warm up epochs')
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")

    parser.add_argument('--print-freq', default=64, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./logs', help='path to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        '--cos',
        action='store_true',
        help='use CosineAnnealing learning rate scheduler'
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use AMP training')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--tb', action='store_true', default=True,
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--adam', action='store_true', default=True, help='Use Adam')
    parser.add_argument('--adamw', action='store_true', help='Use AdamW')
    parser.add_argument('--beta1', default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for Adam')
    parser.add_argument('--dvsaug', action='store_true', default=True, help='Use DVS Data Augumentation')
    parser.add_argument('--random', action='store_true', help='do not fix state')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arg = parse_args()
    main(arg)
