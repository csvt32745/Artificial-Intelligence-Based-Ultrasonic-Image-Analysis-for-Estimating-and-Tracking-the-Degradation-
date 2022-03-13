import numpy as np
from sklearn.metrics import fbeta_score
import os
import cv2
from tqdm import tqdm
import torch

class Scorer():
    ''' Compute the score of masks '''
    def __init__(self, config=None):
        self.predict = []
        self.label = []

        self.boundary_pred = []
        self.boundary_label = []
        boundary_width = 10 if config is None or config.boundary_pixel == 0 else config.boundary_pixel
        boundary_width = boundary_width*2+1
        self.boundary_kernel = np.ones((boundary_width, boundary_width))

    def add(self, predict, label, return_score=False):
        predict = predict.numpy() if isinstance(predict, torch.Tensor) else np.array(predict)
        label = label.numpy() if isinstance(label, torch.Tensor) else np.array(label)

        self.predict.append(predict)
        self.label.append(label)
        self.boundary_pred.append(self.mask_to_boundary(predict))
        self.boundary_label.append(self.mask_to_boundary(label))
        # print(predict.shape, self.boundary_pred[-1].shape)
        # print(label.shape, self.boundary_label[-1].shape)
        if return_score:
            return self._compute_all(predict, label, self.boundary_pred[-1], self.boundary_label[-1])
    
    def mask_to_boundary(self, mask):
        boundary = mask.astype(np.uint8)
        boundary = np.stack([cv2.erode(m, self.boundary_kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1) for m in boundary])
        return mask-boundary

    def _f1(self, predict, label, e = 1):
        temp_predict = np.array(predict).flatten()
        temp_GT = np.array(label).flatten()
        tp = np.sum((temp_predict == 1) * (temp_GT == 1))
        fp = np.sum((temp_predict == 1) * (temp_GT == 0))
        fn = np.sum((temp_predict == 0) * (temp_GT == 1))
        precision = tp / (tp+fp+e)
        recall = tp / (tp+fn+e)
        return 2*precision*recall/(precision+recall)

    def f1(self, e = 1):
        return self._f1(self.predict, self.label, e)

    @staticmethod
    def compute_iou(x, y, e = 1):
        tp_fp = np.sum(x)
        tp_fn = np.sum(y)
        tp = np.sum(x * y)
        iou = (tp + e) / (tp_fp + tp_fn - tp + e)
        return iou

    def iou(self, e = 1):
        temp_predict = np.array(self.predict).flatten()
        temp_GT = np.array(self.label).flatten()
        return self.compute_iou(temp_predict, temp_GT, e)
    
    def _biou(self, predict, label, boundary_pred, boundary_label, e=1):
        temp_predict = np.array(predict).flatten()
        temp_GT = np.array(label).flatten()
        temp_mask_pred = np.array(boundary_pred).flatten()
        temp_mask_label = np.array(boundary_label).flatten()
        
        biou = self.compute_iou(temp_predict*temp_mask_pred, temp_GT*temp_mask_label, e)
        iou = self.compute_iou(temp_predict, temp_GT, e)
        return iou, biou
    
    def biou(self, e=1):
        return self._biou(self.predict, self.label, self.boundary_pred, self.boundary_label)

    def _compute_all(self, predict, label, boundary_pred, boundary_label, e=1):
        iou, biou = self._biou(predict, label, boundary_pred, boundary_label, e)
        f1 = self._f1(predict, label, e)
        return {
            'IoU': iou, 
            'BIoU': biou,
            'F1': f1
            }
        
    def compute_all(self, e = 1):
        ''' return a dict of losses '''
        return self._compute_all(self.predict, self.label, self.boundary_pred, self.boundary_label, e)

    def clear(self):
        self.predict.clear()
        self.label.clear()
        self.boundary_pred.clear()
        self.boundary_label.clear()

    def clear_except_last(self):
        self.predict = [self.predict[-1]]
        self.label = [self.label[-1]]
        self.boundary_pred = [self.boundary_pred[-1]]
        self.boundary_label = [self.boundary_label[-1]]
    
    def concat(self, other, drop_last=False):
        right = -1 if drop_last else None
        self.predict += other.predict[:right]
        self.label += other.label[:right]
        self.boundary_pred += other.boundary_pred[:right]
        self.boundary_label += other.boundary_label[:right]

class Losser():
    def __init__(self):
        self.loss = []

    def add(self, loss_item):
        self.loss.append(loss_item)

    def mean(self):
        return sum(self.loss) / len(self.loss)

    def clear(self):
        self.loss.clear()

class DictLosser():
    def __init__(self):
        self.loss_dict = {}
    
    def add(self, loss_item: dict):
        if not self.loss_dict:
            self.loss_dict = { k: [v] for k, v in loss_item.items() }
            return
        for k, v in loss_item.items():
            self.loss_dict[k].append(v)
    
    def mean(self):
        return { k: np.mean(v) for k, v in self.loss_dict.items() }
    
    def clear(self):
        self.loss_dict.clear()