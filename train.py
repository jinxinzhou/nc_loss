'''
Script for training models.
'''

from torch import optim
import torch
import torch.utils.data
import argparse
import torch.backends.cudnn as cudnn
import random
import json
import os
import numpy as np

# Import dataloaders
import Data.cifar10 as cifar10
import Data.cifar100 as cifar100

# Import network models
from Net.resnet import wide_resnet50_cifar
from train_utils import train_single_epoch, val_single_epoch, print_and_save


dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
}


models = {
    'wide_resnet50': wide_resnet50_cifar,
}


def loss_function_save_name(loss_function,
                            rescale=15,
                            gamma=3.0,
                            smoothing=0.05):
    res_dict = {
        'cross_entropy': 'ce',
        'mean_square_error': 'mse_' + str(rescale),
        'focal_loss': 'fl_gamma_' + str(gamma),
        'label_smoothing': 'ls_smooth_' + str(smoothing),
    }
    res_str = res_dict[loss_function]
    return res_str

def set_seed(manualSeed=666):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)


def parseArgs():
    default_dataset = 'cifar100'
    dataset_root = '../data/'
    train_batch_size = 100
    test_batch_size = 100
    learning_rate = 0.1
    momentum = 0.9
    optimiser = "sgd"
    loss = "cross_entropy"
    smoothing = 0.05
    rescale = 15
    gamma = 3
    weight_decay = 5e-4
    log_interval = 50
    save_loc = './'
    model_name = None
    saved_model_name = ""
    load_loc = './'
    model = "wide_resnet50"
    width = 10
    epoch = 400
    save_interval = epoch // 100
    lr_scheduler = 'step'
    seed = 1

    parser = argparse.ArgumentParser(
        description="Training for calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to train on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=True)

    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("--load", action="store_true", dest="load",
                        help="Load from pretrained model")
    parser.set_defaults(load=False)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("-e", type=int, default=epoch, dest="epoch",
                        help='Number of training epochs')
    parser.add_argument("--lr", type=float, default=learning_rate,
                        dest="learning_rate", help='Learning rate')
    parser.add_argument("--mom", type=float, default=momentum,
                        dest="momentum", help='Momentum')
    parser.add_argument("--nesterov", action="store_true", dest="nesterov",
                        help="Whether to use nesterov momentum in SGD")
    parser.set_defaults(nesterov=False)
    parser.add_argument("--decay", type=float, default=weight_decay,
                        dest="weight_decay", help="Weight Decay")
    parser.add_argument("--opt", type=str, default=optimiser,
                        dest="optimiser",
                        help='Choice of optimisation algorithm')

    parser.add_argument("--loss", type=str, default=loss, dest="loss_function",
                        choices=["cross_entropy", "mean_square_error", "focal_loss", "label_smoothing"],
                        help="Loss function to be used for training")
    parser.add_argument("--loss-mean", action="store_true", dest="loss_mean",
                        help="whether to take mean of loss instead of sum to train")
    parser.set_defaults(loss_mean=True)
    parser.add_argument("--rescale", type=float, default=rescale,
                        dest="rescale", help="rescale parameter for mean_square_error")
    parser.add_argument("--smoothing", type=float, default=smoothing,
                        dest="smoothing", help="smoothing for label smoothing components")
    parser.add_argument("--gamma", type=float, default=gamma,
                        dest="gamma", help="Gamma for focal components")

    parser.add_argument("--log-interval", type=int, default=log_interval,
                        dest="log_interval", help="Log Interval on Terminal")
    parser.add_argument("--save-interval", type=int, default=save_interval,
                        dest="save_interval", help="Save Interval on Terminal")
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to export the model')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name",
                        help='name of the model')
    parser.add_argument("--load-path", type=str, default=load_loc,
                        dest="load_loc",
                        help='Path to load the model from')

    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to train')
    parser.add_argument("--width", type=float, default=width, dest="width",
                        help='Width of wide-resnet')
    parser.add_argument("--lr_scheduler", type=str, default=lr_scheduler,
                        dest="lr_scheduler", help="The type of learing rate decay scheduler")

    parser.add_argument("--seed", type=int, default=seed, dest="seed",
                        help='Random seed')

    return parser.parse_args()


if __name__ == "__main__":

    args = parseArgs()
    set_seed(args.seed)

    if not os.path.exists(args.save_loc):
        os.makedirs(args.save_loc)
    logfile = open('%s/train_log.txt' % (args.save_loc), 'w')

    cuda = False
    if (torch.cuda.is_available() and args.gpu):
        cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))


    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    net = models[args.model](num_classes=num_classes, width=args.width)

    # Setting model name
    if args.model_name is None:
        args.model_name = args.model


    if args.gpu is True:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    start_epoch = 0
    num_epochs = args.epoch
    args.save_interval = num_epochs // 100
    if args.load:
        net.load_state_dict(torch.load(args.save_loc + args.saved_model_name))
        start_epoch = int(args.saved_model_name[args.saved_model_name.rfind('_')+1:args.saved_model_name.rfind('.model')])

    if args.optimiser == "sgd":
        opt_params = net.parameters()
        optimizer = optim.SGD(opt_params,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    elif args.optimiser == "adam":
        opt_params = net.parameters()
        optimizer = optim.Adam(opt_params,
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    if args.lr_scheduler == 'step':
        first_milestone, second_milestone = int(3/7*num_epochs), int(5/7*num_epochs)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[first_milestone, second_milestone], gamma=0.1)
    elif args.lr_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs+1)

    train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
        batch_size=args.train_batch_size,
        augment=args.data_aug,
        random_seed=1,
        pin_memory=args.gpu
    )

    training_set_loss = {}
    val_set_loss = {}
    val_set_err = {}

    for epoch in range(0, start_epoch):
        scheduler.step()

    print_and_save('--------------------- Training -------------------------------', logfile)

    best_epoch = 0
    best_val_acc = 0
    for epoch in range(start_epoch, num_epochs):
        scheduler.step()

        train_loss, train_top1, train_top5 = train_single_epoch(epoch,
                                        net,
                                        train_loader,
                                        optimizer,
                                        device,
                                        loss_function=args.loss_function,
                                        gamma=args.gamma,
                                        rescale=args.rescale,
                                        smoothing=args.smoothing,
                                        loss_mean=args.loss_mean,
                                        num_classes=num_classes,
                                        log_interval=args.log_interval,)
        val_loss, val_top1, val_top5 = val_single_epoch(
                                     net,
                                     val_loader,
                                     device,
                                     loss_function=args.loss_function,
                                     gamma=args.gamma,
                                     smoothing=args.smoothing,
                                     rescale=args.rescale,
                                    loss_mean=args.loss_mean,
                                    num_classes=num_classes)

        training_set_loss[epoch] = train_loss
        val_set_loss[epoch] = val_loss
        val_set_err[epoch] = 100 - val_top1

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            best_epoch = epoch
            print('New best error: %.4f' % (100 - best_val_acc))
            save_name = args.save_loc + 'best' + '.model'
            torch.save(net.state_dict(), save_name)

        if (epoch + 1) % args.save_interval == 0:
            save_name = args.save_loc + str(epoch + 1).zfill(3) + '.model'
            torch.save(net.state_dict(), save_name)

        print_and_save('[Epoch: '+str(epoch + 1).zfill(3)+']'+' | Train Loss: %.4f | Train Top1: %.4f |Train top5: %.4f'
                       %(train_loss, train_top1, train_top5) +
                       ' | Val Loss: %.4f | Val Top1: %.4f |Val top5: %.4f'
                       % (val_loss, val_top1, val_top5) +
                       ' | Best val acc: %.4f | Best epoch: %3d' %
                       (best_val_acc, best_epoch+1), logfile)

    logfile.close()

    with open(save_name[:save_name.rfind('_')] + '_train_loss.json', 'a') as f:
        json.dump(training_set_loss, f)

    with open(save_name[:save_name.rfind('_')] + '_val_loss.json', 'a') as fv:
        json.dump(val_set_loss, fv)

    with open(save_name[:save_name.rfind('_')] + '_val_error.json', 'a') as ft:
        json.dump(val_set_err, ft)
