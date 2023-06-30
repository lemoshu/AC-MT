import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.la_heart import (LA_heart, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case

# CL
import cleanlab

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/LA_data_h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA_ACMT_PErr', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet_MTPD', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=8000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu id')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=8,
                    help='labeled data')
parser.add_argument('--total_sample', type=int, default=80,
                    help='total samples')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')

# PD
parser.add_argument('--uncertainty_th', type=float,  default=0.1, help='threshold')

args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
def getPrototype(features, mask, class_confidence):
    fts = F.interpolate(features, size=mask.shape[-3:], mode='trilinear')  
    mask_new = mask.unsqueeze(1) 
    masked_features = torch.mul(fts, mask_new)  
    masked_fts = torch.sum(masked_features*class_confidence, dim=(2, 3, 4)) / ((mask_new*class_confidence).sum(dim=(2, 3, 4)) + 1e-5)  # bs x C
    return masked_fts


def calDist(fts, mask, prototype):
    """
    Calculate the distance between features and prototypes
    """
    fts_adj_size = F.interpolate(fts, size=mask.shape[-3:], mode='trilinear')
    prototype_new = prototype.unsqueeze(2)
    prototype_new = prototype_new.unsqueeze(3)
    prototype_new = prototype_new.unsqueeze(4)
    dist = torch.sum(torch.pow(fts_adj_size - prototype_new, 2), dim=1, keepdim=True)
    return dist

def masked_entropy_loss(p, mask, C=2):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    y1 = mask * y1
    ent = torch.mean(y1)
    return ent

def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    patch_size = args.patch_size
    max_iterations = args.max_iterations
    num_classes = 2

    def create_model(ema=False):
        # Network definition
        net = net_factory_3d(net_type=args.model,
                             in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    db_train = LA_heart(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.total_sample))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            noisy_ema_inputs = unlabeled_volume_batch + noise
            ema_inputs = unlabeled_volume_batch

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_onehot = torch.argmax(outputs_soft, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)
                LQ_fts = ema_model.featuremap_center 
                noisy_ema_output = ema_model(noisy_ema_inputs)
                noisy_ema_output_soft = torch.softmax(noisy_ema_output, dim=1)
                
			
			# P-Err            
            ema_preds = torch.zeros([10, ema_output.shape[0], ema_output.shape[1], ema_output.shape[2], ema_output.shape[3], ema_output.shape[4]]).cuda()
            ema_fts = torch.zeros([10, LQ_fts.shape[0], LQ_fts.shape[1], LQ_fts.shape[2], LQ_fts.shape[3], LQ_fts.shape[4]]).cuda()
            for i in range(10): 
                with torch.no_grad():
                    ema_preds[i,...] = ema_model(unlabeled_volume_batch)
                    ema_fts[i,...] = ema_model.featuremap_center
            ema_preds1 = torch.sigmoid(ema_preds)
            ema_preds = torch.sigmoid(ema_preds/2.0)
            uncertainty_map = torch.std(ema_preds,dim=0) 
            ema_ft = torch.mean(ema_fts,dim=0) 

            # 2 Uncertainty rectified label
            certain_mask = torch.zeros([uncertainty_map.shape[0], uncertainty_map.shape[1], uncertainty_map.shape[2], uncertainty_map.shape[3], uncertainty_map.shape[4]]).cuda()
            certain_mask[uncertainty_map < args.uncertainty_th] = 1.0 
            
            rect_ema_output_soft = certain_mask * ema_output_soft
            rect_ema_output_onehot = torch.argmax(rect_ema_output_soft, dim=1) 

            # 3 Prototype Generation and Distance Calculation
            # obj prototype and dist (ft to obj prototype, pixel-wise)
            obj_confidence = ema_output_soft[:, 1, ...].unsqueeze(1)
            obj_prototype = getPrototype(ema_ft, rect_ema_output_onehot, obj_confidence) 
            distance_f_obj = calDist(ema_ft, rect_ema_output_onehot, obj_prototype) 

            # bg prototype and dist (ft to bg prototype, pixel-wise)
            bg_confidence = ema_output_soft[:, 0, ...].unsqueeze(1)
            rect_bg_ema_output_onehot = (rect_ema_output_onehot == 0)
            bg_prototype = getPrototype(ema_ft, rect_bg_ema_output_onehot, bg_confidence)       
            distance_f_bg = calDist(ema_ft, rect_bg_ema_output_onehot, bg_prototype) 

            # 4 Compare and Generate Selection Mask
            selection_mask_bg = torch.zeros(distance_f_bg.shape).cuda()
            selection_mask_obj = torch.zeros(distance_f_obj.shape).cuda()
            
            selection_mask_bg[distance_f_obj>distance_f_bg] = 1.0
            selection_mask_obj[distance_f_obj<distance_f_bg] = 1.0
            selection_mask = torch.cat((selection_mask_bg, selection_mask_obj), dim=1)
            
            unlabel_labelbatch_two_channel = torch.zeros_like(outputs[args.labeled_bs:]).scatter_(dim=1,index= outputs_onehot[args.labeled_bs:].unsqueeze(dim=1),src=torch.ones_like(outputs[args.labeled_bs:]))
            
            clean_mask = torch.zeros([outputs_onehot[args.labeled_bs:].shape[0], 2, outputs_onehot[args.labeled_bs:].shape[1],  outputs_onehot[args.labeled_bs:].shape[2],  outputs_onehot[args.labeled_bs:].shape[3]]).cuda()
            clean_mask[selection_mask==unlabel_labelbatch_two_channel] = 1.0
            noisy_mask = (clean_mask == 0)                                            
            
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = losses.softmax_mse_loss(outputs[args.labeled_bs:], noisy_ema_output)  # (batch, 2, 112,112,80)
            #consistency_loss = torch.sum(noisy_mask*consistency_dist)/(torch.sum(noisy_mask)+1e-16)
            consistency_loss = torch.sum(noisy_mask*consistency_dist)/(torch.sum(noisy_mask)+1e-16) + masked_entropy_loss(outputs_soft[args.labeled_bs:], clean_mask, C=2) # opt: add entropy reg.
            
            # Supervised loss
            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs][:])
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 500 and iter_num % 50 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="test.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=18, stride_z=4)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
