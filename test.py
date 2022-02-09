import os

import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by, compute_TIE, compute_FH, create_array, pred_coarse_label
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.renet import RENet


def evaluate(epoch, model, loader, args=None, set='val'):
    model.eval()

    loss_meter = Meter()
    acc_meter = Meter()
    coarse_acc_meter = Meter()
    epoch_tie = []
    epoch_fh = []

    label = torch.arange(args.way).repeat(args.query).cuda()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(loader)

    with torch.no_grad():
        for i, (data, labels, train_coarse_labels) in enumerate(tqdm_gen, 1):
            data = data.cuda()
            model.module.mode = 'encoder'
            data = model(data)
            data_shot, data_query = data[:k], data[k:]
            model.module.mode = 'cca'

            logits, coarse_logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
            loss = F.cross_entropy(logits, label)
            acc = compute_accuracy(logits, label)
            #compute TIE
            realy_coarse = train_coarse_labels[k:]
            realy_coarse = realy_coarse.numpy()
            pred_fine = torch.argmax(logits, dim=1)
            pred_fine = pred_fine.cpu().numpy()
            pred_coarse = pred_coarse_label(pred_fine,realy_coarse)
            t_realy_coarse= torch.Tensor(realy_coarse).cuda()
            tree = create_array(realy_coarse)
            tie = compute_TIE(tree, pred_coarse, realy_coarse)
            fh = compute_FH(tree, pred_coarse, realy_coarse)
            tie_temp = tie / len(pred_coarse)
            epoch_tie.append(tie_temp)
            epoch_fh.append(fh)

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(
                f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.tie:{tie_temp:.3f} | avg.fh:{fh:.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval(), np.mean(epoch_tie), np.mean(epoch_fh)


def test_main(model, args):

    ''' load model '''
    model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))
    # model = load_model(model, os.path.join('checkpoints/cifar_fs/1shot-5way/FS_bseline_5w1s', 'max_acc.pth'))
    ''' define test dataset '''
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    ''' evaluate the model with the dataset '''
    _, test_acc, test_ci, test_tie, test_fh= evaluate("best", model, test_loader, args, set='test')
    print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')

    return test_acc, test_ci, test_tie, test_fh


if __name__ == '__main__':
    args = setup_run(arg_mode='test')

    ''' define model '''
    model = RENet(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    test_main(model, args)
