from os import stat
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from types import MethodType
import cv2

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.BCE_loss = nn.BCEWithLogitsLoss(pos_weight = weight)
    def forward(self, inputs, targets, smooth=1):
        BCE = self.BCE_loss(inputs.float(), targets.float())
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class IOUBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, boundary:int = 0):
        super(IOUBCELoss, self).__init__()
        self.BCE_loss = nn.BCEWithLogitsLoss(pos_weight = weight)
        if boundary > 0:
            print("Use BIoU Loss")
            boundary_width = boundary*2+1
            self.kernel = np.ones((boundary_width, boundary_width))
            self.forward = self.boundary_forward
        else:
            self.forward = self.normal_forward

    
    def normal_forward(self, inputs, targets, smooth=1):
        BCE = self.BCE_loss(inputs.float(), targets.float())
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
        IoU = (intersection + smooth)/(union + smooth)
        IoU_loss = 1 - IoU
        return IoU_loss + BCE
    
    def mask_to_boundary(self, mask):
        # mask: binary mask from pytorch
        boundary = mask.cpu().numpy().astype(np.uint8)
        boundary = np.stack([cv2.erode(m, self.kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1) for m in boundary])
        boundary = torch.from_numpy(boundary.astype(np.float32)).cuda()
        return mask-boundary
    
    @staticmethod
    def iou(x, y, smooth):
        intersection = (x * y).sum()
        total = (x + y).sum()
        union = total - intersection
        IoU = (intersection + smooth)/(union + smooth)
        return IoU

    def boundary_forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=1):
        BCE = self.BCE_loss(inputs.float(), targets.float())
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        
        # extract boundary
        boundary_input = self.mask_to_boundary((inputs>0.5).float()).view(-1)
        boundary_target = self.mask_to_boundary(targets).view(-1)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        IoU = self.iou(inputs, targets, smooth)
        BIoU = self.iou(inputs*boundary_input, targets*boundary_target, smooth)
        IoU_loss = 1 - torch.min(BIoU, IoU)
        return IoU_loss + BCE


class Temporal_Loss(nn.Module):
    def __init__(self, size_average=True, weight=None, gamma = 1.0, distance = None, boundary:int = 0):
        super(Temporal_Loss, self).__init__()
        self.BCE_loss = nn.BCEWithLogitsLoss(pos_weight = weight, reduction = 'none')
        self.temporal_reg = nn.MSELoss()
        max_rho = max(distance)
        self.temporal_weight = (1- (torch.abs(torch.FloatTensor(distance)) / (2*max_rho) )) ** gamma
        self.temporal_weight = nn.Parameter(self.temporal_weight)
        self.temporal_mag = 0.1
        
        self.axis = (0, 2, 3) # (b, t, h, w) -> (t)
        if boundary > 0:
            print("Use BIoU Loss")
            boundary_width = boundary*2+1
            self.kernel = np.ones((boundary_width, boundary_width))
            self.forward = self.boundary_forward
        else:
            self.forward = self.normal_forward
    
    def iou(self, x, y, smooth):
        # return (t, )
        intersection = (x * y).sum(axis=self.axis)
        total = (x + y).sum(axis=self.axis)
        union = total - intersection
        IoU = (intersection + smooth)/(union + smooth)
        return IoU

    def mask_to_boundary(self, mask):
        # mask: binary mask from pytorch
        # (b, t, h, w)
        b, t, h, w = mask.shape
        boundary = mask.cpu().numpy().astype(np.uint8).reshape(-1, h, w)
        boundary = np.stack([cv2.erode(m, self.kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1) for m in boundary])
        boundary = torch.from_numpy(boundary.reshape(b, t, h, w).astype(np.float32)).cuda()
        return mask-boundary
    
    def normal_forward(self, inputs, targets, smooth=1):
        # (b, t, h, w)
        targets = targets.float()
        inputs = inputs.float()
        BCE = self.BCE_loss(inputs, targets).mean(axis=self.axis) 

        logits = F.sigmoid(inputs)
        
        IoU = self.iou(logits, targets, smooth)
        IoU_loss = 1 - IoU

        # temporal_reg = self.temporal_reg(logits[:, :-1], logits[:, 1:])
        # total_loss = (self.temporal_weight*(BCE + IoU_loss)).sum() + temporal_reg * self.temporal_mag
        total_loss = (self.temporal_weight*(BCE + IoU_loss)).sum()
        return total_loss
    
    def boundary_forward(self, inputs, targets, smooth=1):
        # (b, t, h, w)
        targets = targets.float()
        inputs = inputs.float()
        BCE = self.BCE_loss(inputs, targets).mean(axis=self.axis) 

        logits = F.sigmoid(inputs)
        boundary_input = self.mask_to_boundary((logits>0.5).float())
        boundary_target = self.mask_to_boundary(targets)

        IoU = self.iou(logits, targets, smooth)
        BIoU = self.iou(logits*boundary_input, targets*boundary_target, smooth)

        IoU_loss = 1 - torch.minimum(IoU, BIoU)

        # temporal_reg = self.temporal_reg(logits[:, :-1], logits[:, 1:])
        # total_loss = (self.temporal_weight*(BCE + IoU_loss)).sum() + temporal_reg * self.temporal_mag
        total_loss = (self.temporal_weight*(BCE + IoU_loss)).sum()
        return total_loss