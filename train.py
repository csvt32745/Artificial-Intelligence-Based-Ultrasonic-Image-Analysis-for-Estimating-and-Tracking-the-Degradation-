import logging
import numpy as np
import os
import torch
import cv2
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import imageio
## need to remove before submit
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import warnings
# import logging
##net work
from train_src.logger import WBLogger
from train_src.train_code import TemporalTrainer, SingleTrainer
from train_src.dataloader import get_loader, get_continuous_loader
from all_model import WHICH_MODEL
import random
## loss
from train_src.loss_func import DiceBCELoss, IOUBCELoss, Temporal_Loss

from config import config_parser_train
  
def main(config):
    warnings.filterwarnings('ignore')
    seed = 1029
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    # parameter setting
    LR = config.learning_rate
    EPOCH = config.epoch
    BATCH_SIZE = config.batch_size
    if config.continuous == 0:
        frame_continue_num = 0
    else:
        frame_continue_num = list(map(int, config.continue_num))
    
    net, model_name = WHICH_MODEL(config, frame_continue_num)
    
    if config.data_parallel == 1:
        net = nn.DataParallel(net)

    now_time = datetime.now().strftime("%Y_%m_%d_%I:%M:%S")
    if not os.path.isdir(config.save_log_path):
        os.makedirs(config.save_log_path)

    if config.continuous == 1:
        log_name = os.path.join(config.save_log_path, now_time+"_"+model_name+"_"+str(frame_continue_num)+"_gamma="+str(config.gamma)+".log")
        train_weight = torch.FloatTensor([10 / 1])
        criterion_single = IOUBCELoss(weight = train_weight, boundary=config.boundary_pixel).cuda()
        criterion_temporal = Temporal_Loss(
            weight = train_weight, gamma = config.gamma,
            distance = frame_continue_num, boundary=config.boundary_pixel).cuda()
    elif config.continuous == 0:
        log_name = os.path.join(config.save_log_path, now_time+"_"+model_name+".log")
        train_weight = torch.FloatTensor([10 / 1]).cuda()
        criterion_single = IOUBCELoss(weight = train_weight, boundary=config.boundary_pixel).cuda()
    
    # print("log_name: ", log_name)
    # logging.basicConfig(level=logging.DEBUG,
    #                     handlers = [logging.FileHandler(log_name, 'w', 'utf-8'), logging.StreamHandler()])
    # msgfmt = '%(name)s: %(asctime)-15s | %(message)s'
    # datfmt = '%H:%M:%S'
    # logging.basicConfig(format=msgfmt, datefmt=datfmt)
    logger = WBLogger(model_name, log_path=log_name, level=logging.DEBUG, config=config)
    logger.info(sys.argv)
    
    net = net.cuda()
    threshold = config.threshold
    best_score = config.best_score
    OPTIMIZER = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr = LR)
    # scheduler = optim.lr_scheduler.MultiStepLR(OPTIMIZER, milestones=[21], gamma = 0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(OPTIMIZER, T_max=config.epoch, eta_min=1e-6, verbose=True)
    
    if config.continuous == 0:
        logger.info("Single image version")
        train_loader = get_loader(image_path = config.train_data_path,
                                batch_size = BATCH_SIZE,
                                mode = 'train',
                                augmentation_prob = config.augmentation_prob,
                                shffule_yn = True)
        valid_loader = get_loader(image_path = config.valid_data_path,
                                batch_size = 1,
                                mode = 'valid',
                                augmentation_prob = 0.,
                                shffule_yn = False)
        test_loader = get_loader(image_path = config.test_data_path,
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False)
        trainer = SingleTrainer(
            config, logger, net, 
            model_name, threshold, best_score, 
            criterion_single, OPTIMIZER, scheduler,
            train_loader, valid_loader, 
            EPOCH, now_time)
        trainer.Train()
    
    elif config.continuous == 1:
        logger.info("Continuous image version")
        train_loader, continue_num = get_continuous_loader(image_path = config.train_data_path, 
                            batch_size = BATCH_SIZE,
                            mode = 'train',
                            augmentation_prob = config.augmentation_prob,
                            shffule_yn = True,
                            continue_num = frame_continue_num)
        valid_loader, continue_num = get_continuous_loader(image_path = config.valid_data_path,
                                batch_size = 1,
                                mode = 'valid',
                                augmentation_prob = 0.,
                                shffule_yn = False,
                                continue_num = frame_continue_num)
        test_loader, continue_num = get_continuous_loader(image_path = config.test_data_path,
                                batch_size = 1,
                                mode = 'test',
                                augmentation_prob = 0.,
                                shffule_yn = False,
                                continue_num = frame_continue_num)
        logger.info("temporal frame: "+str(continue_num))
        trainer = TemporalTrainer(
            config, logger, net, model_name,
            threshold, best_score, 
            criterion_single, criterion_temporal, OPTIMIZER, scheduler, 
            train_loader, valid_loader,
            EPOCH, continue_num, now_time)
        trainer.Train()

if __name__ == "__main__":
    main(config_parser_train())
