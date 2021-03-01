import os
import sys
sys.path.append("../")

from tqdm import tqdm
import argparse
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from Models import UNet
from utils import ramps, losses
from InputPipeline.DataLoader import CreateDataLoader



parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='../Datasets/ProstatePair/', help='root of the data')
parser.add_argument('--phase', type=str, default='train_40', help='train_phase')
parser.add_argument('--unl_folder', type=str, default='train_60', help='unlabled dataset folder')
parser.add_argument('--exp', type=str,  default='UAMT_unlabel', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batchSize', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--nepoch', type=int,  default=50, help='number of total epochs')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type = int, default = 4, help = 'number of classes')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--nThreads', type=int, default=0, help='num of Threads')
parser.add_argument('--lr', type=float, default=1, help='learnig rate')
parser.add_argument('--fineSize', type=int, default=448, help='learnig rate')
parser.add_argument('--randomScale', action='store_false', default=True, help='whether to do random scale before crop')
parser.add_argument('--isTrain', action='store_false', default=True, help='whether in train phase')
parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data argumentation')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

args.device = torch.device("cuda:0" if args.gpu=='1' else "cpu")

# train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batchSize = args.batchSize * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# num_classes = 2
# patch_size = (112, 112, 80)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def featch_data(iterator, dataloader):
    try:
        data = next(iterator)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        iterator = iter(dataloader)
        data = next(iterator)
    return iterator, data


if __name__ == "__main__":

    def create_model(ema=False):
        # Network definition
        net = UNet(n_channels=3, n_classes=args.num_classes, bilinear = True)
        model = net.to(args.device)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model() ## student model
    ema_model = create_model(ema = True) ## teacher model

    ## create dataset
    dataloader_l = CreateDataLoader(args, batchSize = args.batchSize, shuffle = True, fixed = False, dataset = None)
    dataloader_l = dataloader_l.load_data()
    dataloader_unl = CreateDataLoader(args, batchSize = args.batchSize, shuffle = True, fixed = False, dataset = None)
    dataloader_unl = dataloader_unl.load_data()


    ## start to train
    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr =args.lr, momentum = 0.9, weight_decay = 0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    iter_num = 0
    for epoch in range(args.nepoch):
        for j in range(int(len(dataloader_l) * (100 / 80))):
            label_iter = iter(dataloader_l)
            unl_iter = iter(dataloader_unl)
            ## 1. featch label data from data loader
            label_iter, lab = featch_data(label_iter, dataloader_l)
            ## 2. featch unlabel data from data loader
            unl_iter, unl = featch_data(unl_iter, dataloader_unl)

            lab_image = lab['image'].to(args.device)
            lab_label = lab['label'].to(args.device)
            unl_image = unl['image'].to(args.device)


            noise = torch.clamp(torch.randn_like(unl_image) * 0.1, -0.2, 0.2)
            ema_inputs = unl_image + noise

            outputs = model(lab_image) ## outputs logists [N, C, W, H]
            unl_pseu_label = model(unl_image)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)

            loss_seg = F.cross_entropy(outputs, torch.squeeze(lab_label, dim = 1).long())
            outputs_soft = F.softmax(outputs, dim = 1)
            loss_seg_dice = losses.dice_loss(outputs, lab_label)
            supervised_loss = 0.5 * (loss_seg + loss_seg_dice)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = consistency_criterion(unl_pseu_label, ema_output)

            consistency_loss = consistency_weight * torch.sum(consistency_dist)
            loss = supervised_loss + 0.001 * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num) ## update teacher model

            iter_num = iter_num + 1

            if iter_num % 1 == 0:
                print(f"Current iter: {iter_num}, loss: {loss}.")
        torch.save(ema_model.state_dict(), f"TrainedModels/ema_model_{epoch}.pth")
