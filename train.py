import os
import tqdm
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.utils import compute_accuracy, load_model, setup_run, by
from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run, compute_TIE, compute_FH, create_array, pred_coarse_label
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.renet import RENet
from test import test_main, evaluate


c_lambda = 0.4
f_lambda = 1.0 - c_lambda

def train(epoch, model, loader, optimizer, args=None):
    model.train()

    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']

    # label for query set, always in the same pattern
    label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...

    loss_meter = Meter()
    acc_meter = Meter()
    # coarse_metter = Meter()
    epoch_tie = []
    epoch_fh = []

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(train_loader)
    cur_lambda = 0.0
    # for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):
    for i, ((data, train_labels, train_coarse_labels), (data_aux, train_labels_aux, train_coarse_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):
        data, train_labels, train_coarse_labels = data.cuda(), train_labels.cuda(), train_coarse_labels.cuda()
        # data, train_labels = data.cuda(), train_labels.cuda()
        data_aux, train_labels_aux, train_coarse_labels_aux = data_aux.cuda(), train_labels_aux.cuda(), train_coarse_labels_aux.cuda()

        # Forward images (3, 84, 84) -> (C, H, W)
        model.module.mode = 'encoder'
        data = model(data)
        data_aux = model(data_aux)  # I prefer to separate feed-forwarding data and data_aux due to BN

        # loss for batch
        model.module.mode = 'cca'
        data_shot, data_query = data[:k], data[k:]
        logits, absolute_logits, coarse_logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
        epi_loss = F.cross_entropy(logits, label)
        
        
        #compute coarse loss
        coarse_loss = F.cross_entropy(coarse_logits, train_coarse_labels[k:])
        absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])

        # loss for auxiliary batch
        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        loss_aux = F.cross_entropy(logits_aux, train_labels_aux)
        
        loss_aux = loss_aux + f_lambda * absolute_loss + c_lambda * coarse_loss

        loss = args.lamb * epi_loss + loss_aux
        acc = compute_accuracy(logits, label)
        realy_coarse = train_coarse_labels[k:]
        realy_coarse = realy_coarse.cpu().numpy()
        tree = create_array(realy_coarse)
        pred_fine = torch.argmax(logits, dim=1)
        pred_fine = pred_fine.cpu().numpy()
        pred_coarse = pred_coarse_label(pred_fine, realy_coarse)
        tie = compute_TIE(tree,pred_coarse,realy_coarse)
        fh = compute_FH(tree,pred_coarse,realy_coarse)
        tie_temp = tie / len(pred_coarse)
        epoch_tie.append(tie_temp)
        epoch_fh.append(fh)


        loss_meter.update(loss.item())
        acc_meter.update(acc)

        # tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.coarse_acc:{coarse_metter.avg():.3f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')
        tqdm_gen.set_description(
            f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.tie:{tie_temp:.3f} | avg.fh:{fh:.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()
    # print('epoch:', epoch, '  my_lambda', cur_lambda)
    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval(),np.mean(epoch_tie), np.mean(epoch_fh)


def train_main(args):
    Dataset = dataset_builder(args)

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, len(trainset.data) // args.batch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    ''' fix val set for all epochs '''
    val_loader = [x for x in val_loader]

    set_seed(args.seed)
    model = RENet(args).cuda()
    # model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))
    model = nn.DataParallel(model, device_ids=args.device_ids)

    if not args.no_wandb:
        wandb.watch(model)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    max_acc, max_epoch,min_tie, max_fh = 0.0, 0,0.0,0.0
    set_seed(args.seed)

    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()

        train_loss, train_acc, _, train_tie, train_fh = train(epoch, model, train_loaders, optimizer, args)
        val_loss, val_acc, _, val_tie, val_fh = evaluate(epoch, model, val_loader, args, set='val')

        if not args.no_wandb:
            # wandb.log({'train/loss': train_loss, 'train/acc': train_acc, 'train/coarse_acc': train_coarse_acc,'val/loss': val_loss, 'val/acc': val_acc}, step=epoch)
            wandb.log({'train/loss': train_loss, 'train/acc': train_acc, 'train/tie': train_tie, 'train/fh': train_fh, 'val/loss': val_loss, 'val/acc': val_acc, 'val/tie': val_tie, 'val/fh': val_fh,},
                      step=epoch)

        if val_acc > max_acc:
            print(f'[ log ] *********A better model is found ({val_acc:.3f}) *********')
            max_acc, max_epoch,min_tie, max_fh  = val_acc, epoch,val_tie, val_fh
            print(f'[ log ] *********The in_tie ({min_tie:.3f}), min_fh ({max_fh:.3f}) *********')
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))

        if args.save_all:
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, f'epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'optimizer_epoch_{epoch}.pth'))

        epoch_time = time.time() - start_time
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

        lr_scheduler.step()

    return model


if __name__ == '__main__':
    args = setup_run(arg_mode='train')

    model = train_main(args)
    test_acc, test_ci, test_tie, test_fh= test_main(model, args)

    if not args.no_wandb:

        wandb.log(
            {'test/acc': test_acc, 'test/confidence_interval': test_ci, 'test/tie': test_tie, 'test/fh': test_fh })