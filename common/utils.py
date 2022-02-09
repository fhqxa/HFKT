import os
import wandb
import torch
import pprint
import random
import argparse
import numpy as np
from termcolor import colored


def setup_run(arg_mode='train'):
    args = parse_args(arg_mode=arg_mode)
    pprint(vars(args))

    torch.set_printoptions(linewidth=100)
    args.num_gpu = set_gpu(args)
    args.device_ids = None if args.gpu == '-1' else list(range(args.num_gpu))
    args.save_path = os.path.join(f'checkpoints/{args.dataset}/{args.shot}shot-{args.way}way/', args.extra_dir)
    ensure_path(args.save_path)

    if not args.no_wandb:
        wandb.init(project=f'renet-{args.dataset}-{args.way}w{args.shot}s',
                   config=args,
                   save_code=True,
                   name=args.extra_dir)

    if args.dataset == 'miniimagenet':
        args.num_class = 64
    elif args.dataset == 'cub':
        args.num_class = 100
    elif args.dataset == 'cifar_fc':
        args.num_class = 60
        args.num_fine_class = 60
        args.num_coarse_class = 12
    elif args.dataset == 'tieredimagenet':
        args.num_class = 351
        args.num_fine_class = 351
        args.num_coarse_class = 20
    elif args.dataset == 'cifar_fs':
        args.num_class = 64
        args.num_fine_class = 64
        args.num_coarse_class = 20
    elif args.dataset == 'cars':
        args.num_class = 130
    elif args.dataset == 'dogs':
        args.num_class = 70
    elif args.dataset == 'voc':
        args.num_class = 7

    return args


def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)

def compute_accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.float).mean().item() * 100.


#YulinSu TIE,FH
def tree_ANcestor(tr, nd):
    #  tree: 二维数组   node：节点  默认根节点为0
    A = [nd] # 存储
    # print(nd-1)
    nd_ = tr[nd - 1]  # python 从 0 开始算，而结点从 1 开始算

    while nd_ > 0:
        A.append(nd_)
        nd_ = tr[nd_ - 1]  # 找父结点

    return A
# TIE
def compute_TIE(tr, p_nd, r_nd):
    TIE = 0
    for i in range(len(p_nd)):
        r_anc = tree_ANcestor(tr, r_nd[i])  # 真实标签的父结点
        p_anc = tree_ANcestor(tr, p_nd[i])  # 预测标签的父结点
        b = list(set(r_anc).difference(set(p_anc)))  # 取 r_anc 与 p_anc 的差集
        c = list(set(p_anc).difference(set(r_anc)))  # 取 p_anc 与 r_anc 的差集
        TIE = TIE + len(b + c)
    # r_anc = tree_ANcestor(tr, r_nd)  # 真实标签的父结点
    # p_anc = tree_ANcestor(tr, p_nd)  # 预测标签的父结点
    # b = list(set(r_anc).difference(set(p_anc)))  # 取 r_anc 与 p_anc 的差集
    # c = list(set(p_anc).difference(set(r_anc)))  # 取 p_anc 与 r_anc 的差集
    # TIE = len(b + c)
    return TIE
# create tree_array
def create_array(coarse_list):
    num_coarse = max(coarse_list) + 1
    tree = [0]
    for i in range(num_coarse):
        tree.append(1)

    for x in coarse_list:
        tree.append(x + 2)

    return tree
# FH
def compute_FH(tr, p_nd, r_nd):
    sum_PH, sum_RH, sum_FH = 0, 0, 0
    length = len(p_nd)
    for i in range(length):
        r_anc = tree_ANcestor(tr, r_nd[i])  # 真实标签的父结点
        p_anc = tree_ANcestor(tr, p_nd[i])  # 预测标签的父结点
        b = [x for x in r_anc if x in p_anc]  # 取 r_anc 与 p_anc 的交集

        PH = len(b) / len(p_anc)
        RH = len(b) / len(r_anc)
        FH = 2 * PH * RH / (PH + RH)

        sum_PH = sum_PH + PH
        sum_RH = sum_RH + RH
        sum_FH = sum_FH + FH

    PH = sum_PH / length
    RH = sum_RH / length
    FH = sum_FH / length

    return FH    #, PH, RH
#pred coarse
def pred_coarse_label(pred_fine_label,sample_coarse_label):
    coarse_labels = []
    for x in pred_fine_label:
         coarse_labels.append(sample_coarse_label[x])
    return coarse_labels

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def load_model(model, dir):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(dir)['params']

    if pretrained_dict.keys() == model_dict.keys():  # load from a parallel meta-trained model and all keys match
        print('all state_dict keys match, loading model from :', dir)
        model.load_state_dict(pretrained_dict)
    else:
        print('loading model from :', dir)
        if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
            if 'module' in list(pretrained_dict.keys())[0]:
                pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # load from a pretrained model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
        model.load_state_dict(model_dict)

    return model


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def detect_grad_nan(model):
    for param in model.parameters():
        if (param.grad != param.grad).float().sum() != 0:  # nan detected
            param.grad.zero_()


def by(s):
    '''
    :param s: str
    :type s: str
    :return: bold face yellow str
    :rtype: str
    '''
    bold = '\033[1m' + f'{s:.3f}' + '\033[0m'
    yellow = colored(bold, 'yellow')
    return yellow


def parse_args(arg_mode):
    parser = argparse.ArgumentParser(description='Relational Embedding for Few-Shot Classification (ICCV 2021)')    #Hierarchical few-shot learning with attention feature fusion

    ''' about dataset '''
    parser.add_argument('-dataset', type=str, default='tieredimagenet',
                        choices=['miniimagenet', 'cub', 'tieredimagenet', 'cifar_fs', 'cifar_fc', 'voc'])                     #update   source:miniimagenet
    parser.add_argument('-data_dir', type=str, default='/home/grcwoods/WZP/Datas/tirered_imagenet', help='dir of datasets')   #update   source:datasets

    ''' about training specs '''
    parser.add_argument('-batch', type=int, default=128, help='auxiliary batch size')                                         #update   source:128
    parser.add_argument('-temperature', type=float, default=5.0, metavar='tau', help='temperature for metric-based loss')     #update   source:0.2
    parser.add_argument('-lamb', type=float, default=0.25, metavar='lambda', help='loss balancing term')

    ''' about training schedules '''
    parser.add_argument('-max_epoch', type=int, default=80, help='max epoch to run')
    parser.add_argument('-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.05, help='learning rate decay factor')
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70], help='milestones for MultiStepLR')            #update     source:[60,70]
    parser.add_argument('-save_all', action='store_true', help='save models on each epoch')

    ''' about few-shot episodes '''
    parser.add_argument('-way', type=int, default=5, metavar='N', help='number of few-shot classes')
    parser.add_argument('-shot', type=int, default=1, metavar='K', help='number of shots')
    parser.add_argument('-query', type=int, default=15, help='number of query image per class')
    parser.add_argument('-val_episode', type=int, default=200, help='number of validation episode')
    parser.add_argument('-test_episode', type=int, default=2000, help='number of testing episodes after training')

    ''' about SCR '''
    parser.add_argument('-self_method', type=str, default='scr')

    ''' about CCA '''
    parser.add_argument('-temperature_attn', type=float, default=5.0, metavar='gamma', help='temperature for softmax in computing cross-attention')

    ''' about env '''
    parser.add_argument('-gpu', default='0', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('-extra_dir', type=str, default='Test_addcoarse', help='extra dir name added to checkpoint dir')         #update
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-no_wandb', action='store_true', help='not plotting learning curve on wandb',
                        default=arg_mode == 'test')  # train: enable logging / test: disable logging
    args = parser.parse_args()
    return args
