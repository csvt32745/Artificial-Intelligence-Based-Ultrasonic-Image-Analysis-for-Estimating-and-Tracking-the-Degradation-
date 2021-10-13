import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from other.visualization import Vis_Input_Merge_Heat

from train_src.dataloader import denormalize_img_rgb
from train_src.logger import WBLogger
from train_src.Score import DictLosser, Scorer, Losser
import ipdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from abc import ABC, abstractclassmethod

class BaseTrainer(ABC):
    def __init__(self, 
        config, logger: WBLogger , net: nn.Module, model_name, 
        threshold, best_score, 
        optimizer: optim.Optimizer, scheduler, 
        train_loader, valid_loader, 
        epoch, now_time, log_interval=100):
        
        self.logger = logger
        self.config = config
        self.model_name = model_name
        self.now_time = now_time
        self.epoch = epoch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net
        self.seg_threshold = threshold
        self.best_score = best_score
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.scorer = Scorer(config)
        self.losser = DictLosser()
        self.image_buffer = {}
        

        self.iteration = 0
        self.logging_interval = log_interval
        self.net_save_path = os.path.join(
            self.config.save_model_path, self.model_name, self.now_time+self.model_name)
    
    
    @abstractclassmethod
    def UnpackInputData(self, x):
        ''' 
        Unpack data from loader into X(input), Y(GT)
        output should be tuples
        '''
        pass
    
    @abstractclassmethod
    def Criterion(self, output, GT):
        ''' 
        calculate losses
        return 
            total_loss_tensor (for backward),
            losses_dict (for logging)
        '''
        pass
        
    @abstractclassmethod
    def RecordMetrics(self, epoch, x, GT, output, loss_dict, isTraining):
        '''
        record/collect data like images and metrics
        '''
        pass

    def RecordMetricsSummary(self, epoch, isTraining):
        '''
        record/collect data right before logging (after a epoch/interval)
        '''
        pass

    def SaveModel(self, save_path, file_name):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        best_net = self.net.state_dict()
        torch.save(best_net, os.path.join(save_path, file_name))
        self.logger.info("Model save in "+ save_path)

    def IfSaveModel(self, epoch):
        iou = self.scorer.iou()
        if iou >= self.best_score or (epoch+1) % 5 == 0 or (epoch+1) <= 6:
            self.best_score = iou
            file_name = f"Epoch={epoch+1}_Score={self.best_score:.4f}.pt"
            self.SaveModel(self.net_save_path, file_name)

    def CollectMetrics(self):
        metric_dict = self.scorer.compute_all()
        metric_dict.update(self.losser.mean())
        return metric_dict

    def Train(self):
        for epoch in range(self.epoch):
            self.EpochTrain(epoch)
            self.EpochValidation(epoch)
            self.IfSaveModel(epoch)
    
    def EpochTrain(self, epoch):
        self.net.train()
        self.scorer.clear()
        self.losser.clear()

        for i, x in enumerate(tqdm(self.train_loader, position=0, leave=True, dynamic_ncols=True)):
            x, GT = self.UnpackInputData(x)
            output = self.net(*x) # untuple x
            total_loss, loss_dict = self.Criterion(output, GT)

            self.optimizer.zero_grad() 
            total_loss.backward()
            self.optimizer.step()

            self.RecordMetrics(epoch, x, GT, output, loss_dict, isTraining=True)
            self.iteration += 1
            
            if (i+1) % self.logging_interval == 0:
                self.RecordMetricsSummary(epoch, isTraining=True)
                self.logger.LogTrainingDB(
                    epoch, self.epoch, i, len(self.train_loader),
                    self.iteration, self.CollectMetrics(), self.image_buffer)
                self.scorer.clear()
                self.losser.clear()
                self.image_buffer.clear()
            
        self.scheduler.step()

    @torch.no_grad()
    def EpochValidation(self, epoch):
        self.net.eval()
        self.scorer.clear()
        self.losser.clear()
        self.image_buffer.clear()

        for i, x in enumerate(tqdm(self.valid_loader, position=0, leave=True)):
            x, GT = self.UnpackInputData(x)
            output = self.net(*x) # untuple x
            total_loss, loss_dict = self.Criterion(output, GT)
            self.RecordMetrics(epoch, x, GT, output, loss_dict, isTraining=False)
            
            
        
        self.RecordMetricsSummary(epoch, isTraining=False)
        self.logger.LogValidationDB(epoch, self.epoch, self.iteration, self.CollectMetrics(), self.image_buffer)

class TemporalTrainer(BaseTrainer):
    def __init__(self, config, logger, net, model_name, threshold, best_score, criterion_single, criterion_temporal, optimizer, scheduler, train_loader, valid_loader, epoch, continue_num, now_time):
        super().__init__(
            config, logger, net, 
            model_name, threshold, best_score, 
            optimizer, scheduler, train_loader, valid_loader, 
            epoch, now_time)

        self.criterion_single = criterion_single
        self.criterion_temporal = criterion_temporal
        self.continue_num = continue_num
        self.seq_len = len(continue_num)
        self.net_save_path = os.path.join(
            self.config.save_model_path, self.model_name, self.now_time+str(self.continue_num))
        self.cur_sample_name = ""

    def UnpackInputData(self, x):
        ''' 
        Unpack data from loader into X(input), Y(GT)
        output should be tuples
        '''
        self.cur_sample_name, image_list, mask_list = x
        self.cur_sample_name = self.cur_sample_name[1 + self.seq_len//2][0]
        
        return \
            (image_list[:, :1], image_list[:,1:]),\
            (mask_list[:,:1].squeeze(dim = 1).cuda(), mask_list[:,1:].cuda())
        pass
    
    def Criterion(self, output, GT):
        ''' 
        calculate losses
        return 
            total_loss_tensor (for backward),
            losses_dict (for logging)
        '''
        temporal_output, output = output
        GT, temporal_GT = GT

        output = output.squeeze(dim = 1) # need to delete?
        loss = self.criterion_single(output, GT.float())

        if self.config.w_T_LOSS == 1:
            pn_loss = self.criterion_temporal(temporal_output, temporal_GT)
            return (
                loss + pn_loss,
                {
                    'SingleLoss': loss.item(),
                    'TemporalLoss': pn_loss.item()
                }
            )
        else:
            pn_loss = torch.tensor(0).cuda()
            return (
                loss,
                { 'SingleLoss': loss.item(), }
            )
        
    def RecordMetrics(self, epoch, x, GT, output, loss_dict, isTraining, img_name=''):
        '''
        record/collect data like images and metrics
        '''
        # Metrics
        self.losser.add(loss_dict)
        GT = GT[0].cpu()
        output = F.sigmoid(output[1])
        SR = torch.where(output > self.seg_threshold, 1, 0).cpu()
        self.scorer.add(SR, GT)

        # Images
        if not isTraining:
            loss = loss_dict['TemporalLoss']
            worst_loss = self.image_buffer.get('WorstSample')
            if (not worst_loss) or (loss > worst_loss[0]):
                self.image_buffer['WorstSample'] = (
                    loss,
                    x[0].cpu().numpy().squeeze().transpose(1, 2, 0),
                    SR.numpy().squeeze(),
                    GT.numpy().squeeze(),
                    output.cpu().numpy().squeeze(),
                    self.cur_sample_name
                )


    def RecordMetricsSummary(self, epoch, isTraining):
        # Images
        if not isTraining:
            imgs = self.image_buffer['WorstSample']
            name = f"Epoch: {epoch}, Loss: {imgs[0]:.4f}, {imgs[-1]}"
            self.image_buffer['WorstSample'] = Vis_Input_Merge_Heat(*(imgs[1:-1]), name)
            

class SingleTrainer(BaseTrainer):
    def __init__(self, config, logger, net: nn.Module, model_name, threshold, best_score, criterion, optimizer: optim.Optimizer, scheduler, train_loader, valid_loader, epoch, now_time, log_interval=100):
        super().__init__(
            config, logger, net, 
            model_name, threshold, best_score, 
            optimizer, scheduler, 
            train_loader, valid_loader, 
            epoch, now_time, log_interval=log_interval)
        
        self.criterion = criterion

    def UnpackInputData(self, x):
        ''' 
        Unpack data from loader into X(input), Y(GT)
        output should be tuples
        '''
        return (x[0],), (x[1].cuda(),)
    
    def Criterion(self, output, GT):
        ''' 
        calculate losses
        return 
            total_loss_tensor (for backward),
            losses_dict (for logging)
        '''
        loss = self.criterion(output.squeeze(1), GT[0].float())
        return (
            loss,
            { 'Loss': loss.item(), }
        )
        
    def RecordMetrics(self, epoch, x, GT, output, loss_dict):
        '''
        record/collect data like images and metrics
        '''
        self.losser.add(loss_dict)
        SR = torch.where(F.sigmoid(output) > self.seg_threshold, 1, 0).cpu()
        self.scorer.add(SR, GT[0].cpu())

