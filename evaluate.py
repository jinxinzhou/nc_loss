import os
import sys
import numpy as np
import torch
import random
import pickle
import argparse
import scipy.linalg as scilin
from torch import nn
import torch.backends.cudnn as cudnn

# Import dataloaders
import Data.cifar10 as cifar10
import Data.cifar100 as cifar100

# Import network architectures
from Net.resnet import wide_resnet50_cifar


from train_utils import AverageMeter, compute_accuracy, print_and_save

# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
}

# Mapping model name to model function
models = {
    'wide_resnet50': wide_resnet50_cifar,
}

def set_seed(manualSeed=666):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)


def split_array(input_array, batchsize=512):
    input_size = input_array.shape[0]
    num_splits, res_splits = input_size // batchsize, input_size % batchsize
    output_array_list = list()
    if res_splits == 0:
        output_array_list = np.split(input_array, num_splits, axis=0)
    else:
        for i in range(num_splits):
            output_array_list.append(input_array[i * batchsize:(i + 1) * batchsize])

        output_array_list.append(input_array[num_splits * batchsize:])

    return output_array_list


def compute_info(model, dataloader, device):
    num_data = 0
    mu_G = 0
    mu_c_dict = dict()
    num_class_dict = dict()
    before_class_dict = dict()
    after_class_dict = dict()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            logits, features = model(inputs)

        mu_G += torch.sum(features, dim=0)

        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict:
                mu_c_dict[y] = features[b, :]
                before_class_dict[y] = [features[b, :].detach().cpu().numpy()]
                after_class_dict[y] = [logits[b, :].detach().cpu().numpy()]
                num_class_dict[y] = 1
            else:
                mu_c_dict[y] += features[b, :]
                before_class_dict[y].append(features[b, :].detach().cpu().numpy())
                after_class_dict[y].append(logits[b, :].detach().cpu().numpy())
                num_class_dict[y] = num_class_dict[y] + 1

        num_data += targets.shape[0]

        prec1, prec5 = compute_accuracy(logits.data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    mu_G /= num_data
    for i in range(len(mu_c_dict.keys())):
        mu_c_dict[i] /= num_class_dict[i]

    return mu_G, mu_c_dict, before_class_dict, after_class_dict, top1.avg, top5.avg


def compute_Sigma_W(before_class_dict, mu_c_dict, device, batchsize=100):
    num_data = 0
    Sigma_W = 0

    for target in before_class_dict.keys():
        class_feature_list = split_array(np.array(before_class_dict[target]), batchsize=batchsize)
        for features in class_feature_list:
            features = torch.from_numpy(features).to(device)
            Sigma_W_batch = (features - mu_c_dict[target].unsqueeze(0)).unsqueeze(2) @ (
                    features - mu_c_dict[target].unsqueeze(0)).unsqueeze(1)
            Sigma_W += torch.sum(Sigma_W_batch, dim=0)
            num_data += features.shape[0]

    Sigma_W /= num_data
    return Sigma_W.detach().cpu().numpy()


def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()


def compute_ETF(W, device):
    K = W.shape[0]
    W = W - torch.mean(W, dim=0, keepdim=True)
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, mu_c_dict, mu_G, device):
    K = len(mu_c_dict)
    H = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        H[:, i] = mu_c_dict[i] - mu_G

    W = W - torch.mean(W, dim=0, keepdim=True)
    WH = torch.mm(W, H.to(device))
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device)

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item(), H


def compute_Wh_b_relation(W, mu_G, b, device):
    Wh = torch.mv(W, mu_G.to(device))
    res_b = torch.norm(Wh + b, p='fro')
    return res_b.detach().cpu().numpy().item()


def parseArgs():
    default_dataset = 'cifar100'
    dataset_root = '../data'
    model = 'wide_resnet50'
    width = 2
    epoch = 400
    save_interval = epoch // 100
    save_loc = './'
    model_name = None
    train_batch_size = 128
    test_batch_size = 128
    seed = 1

    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to test on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name", help='name of the model')
    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to test')
    parser.add_argument("--width", type=float, default=width, dest="width",
                        help='Width of wide-resnet')
    parser.add_argument("--save-interval", type=int, default=save_interval,
                        dest="save_interval", help="Save Interval on Terminal")
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to import the model')
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-da", action="store_true", dest="data_aug",
                        help="Using data augmentation")
    parser.set_defaults(data_aug=True)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("-log", action="store_true", dest="log",
                        help="whether to print log data")
    parser.add_argument("-e", type=int, default=epoch, dest="epoch",
                        help='Number of training epochs')
    parser.add_argument("--seed", type=int, default=seed, dest="seed",
                        help='Random seed')

    return parser.parse_args()


def get_logits_labels(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            logits, _ = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    return logits, labels


if __name__ == "__main__":

    # Checking if GPU is available
    cuda = False
    if (torch.cuda.is_available()):
        cuda = True

    args = parseArgs()

    # Setting additional parameters
    set_seed(args.seed)
    device = torch.device("cuda" if cuda else "cpu")

    if args.model_name is None:
        args.model_name = args.model

    dataset = args.dataset
    dataset_root = args.dataset_root
    model_name = args.model_name
    save_loc = args.save_loc
    #     saved_model_name = args.saved_model_name
    args.save_interval = args.epoch // 100

    # Taking input for the dataset
    num_classes = dataset_num_classes[dataset]
    train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
        batch_size=args.train_batch_size,
        augment=args.data_aug,
        random_seed=1,
        pin_memory=args.gpu
    )

    test_loader = dataset_loader[args.dataset].get_test_loader(
        batch_size=args.test_batch_size,
        pin_memory=args.gpu
    )

    model = models[model_name]

    net = model(num_classes=num_classes, width=args.width)
    net.cuda()

    info_dict = {
        'collapse_metric': [],
        'ETF_metric': [],
        'WH_relation_metric': [],
        'Wh_b_relation_metric': [],
        'avg_cos_margin': [],
        'all_cos_margin': [],
        'W': [],
        'b': [],
        'mu_G_train': [],
        'mu_G_val': [],
        'mu_G_test': [],
        'mu_c_dict_train': [],
        'mu_c_dict_val': [],
        'mu_c_dict_test': [],
        'train_acc1': [],
        'train_acc5': [],
        'val_acc1': [],
        'val_acc5': [],
        'test_acc1': [],
        'test_acc5': [],
        'train_best_top1': None,
        'val_best_top1': None,
        'best_val_test_top1': None
    }

    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    logfile = open('%s/test_log.txt' % (args.save_loc), 'w')

    for i in range(args.epoch):
        if (i + 1) % args.save_interval == 0:
            net.load_state_dict(torch.load(args.save_loc + str(i + 1).zfill(3) + '.model'))
            net.eval()

            b = None
            for n, p in net.named_parameters():
                if 'fc.weight' in n:
                    W = p
                if 'fc.bias' in n:
                    b = p
            if b is None:
                b = torch.zeros((W.shape[0],), device=device)

            mu_G_train, mu_c_dict_train, before_class_dict_train, after_class_dict_train, \
            train_acc1, train_acc5 = compute_info(net, train_loader, device)

            mu_G_val, mu_c_dict_val, before_class_dict_val, after_class_dict_val, \
            val_acc1, val_acc5 = compute_info(net, val_loader, device)

            mu_G_test, mu_c_dict_test, before_class_dict_test, after_class_dict_test, \
            test_acc1, test_acc5 = compute_info(net, test_loader, device)

            Sigma_W = compute_Sigma_W(before_class_dict_train, mu_c_dict_train, device)
            Sigma_B = compute_Sigma_B(mu_c_dict_train, mu_G_train)

            collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
            ETF_metric = compute_ETF(W, device)

            WH_relation_metric, H = compute_W_H_relation(W, mu_c_dict_train, mu_G_train, device)
            Wh_b_relation_metric = compute_Wh_b_relation(W, mu_G_train, b, device)

            info_dict['collapse_metric'].append(collapse_metric)
            info_dict['ETF_metric'].append(ETF_metric)
            info_dict['WH_relation_metric'].append(WH_relation_metric)
            info_dict['Wh_b_relation_metric'].append(Wh_b_relation_metric)

            info_dict['train_acc1'].append(train_acc1)
            info_dict['train_acc5'].append(train_acc5)

            info_dict['val_acc1'].append(val_acc1)
            info_dict['val_acc5'].append(val_acc5)

            info_dict['test_acc1'].append(test_acc1)
            info_dict['test_acc5'].append(test_acc5)

            print_and_save(
                '[epoch: %d] | train top1: %.4f | train top5: %.4f | val top1: %.4f | val top5: %.4f | test top1: %.4f | test top5: %.4f ' %
                (i + 1, train_acc1, train_acc5, val_acc1, val_acc5, test_acc1, test_acc5), logfile)

    net.load_state_dict(torch.load(args.save_loc + 'best' + '.model'))
    net.eval()

    b = None
    for n, p in net.named_parameters():
        if 'fc.weight' in n:
            W = p
        if 'fc.bias' in n:
            b = p
    if b is None:
        b = torch.zeros((W.shape[0],), device=device)

    mu_G_train, mu_c_dict_train, before_class_dict_train, after_class_dict_train, \
            train_acc1, train_acc5 = compute_info(net, train_loader, device)

    mu_G_val, mu_c_dict_val, before_class_dict_val, after_class_dict_val, \
            val_acc1, val_acc5 = compute_info(net, val_loader, device)

    mu_G_test, mu_c_dict_test, before_class_dict_test, after_class_dict_test, \
            test_acc1, test_acc5 = compute_info(net, test_loader, device)

    info_dict['train_best_top1'] = train_acc1
    info_dict['val_best_top1'] = val_acc1
    info_dict['best_val_test_top1'] = test_acc1

    print_and_save(
        '[Best] | Train Top1: %.4f | Val Top1: %.4f | Test Top1: %.4f ' %
        (train_acc1, val_acc1, test_acc1), logfile)


    logfile.close()
    with open(args.save_loc + '/info.pkl', 'wb') as f:
        pickle.dump(info_dict, f)
