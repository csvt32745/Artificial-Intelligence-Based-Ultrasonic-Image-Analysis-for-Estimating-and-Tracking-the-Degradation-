import numpy as np
from sklearn.metrics import fbeta_score
import os
import cv2
from tqdm import tqdm

class Scorer():
    ''' Compute the score of masks '''
    def __init__(self, config):
        self.predict = []
        self.label = []

        self.boundary_pred = []
        self.boundary_label = []
        if config.boundary_pixel == 0:
            config.boundary_pixel = 10
        boundary_width = config.boundary_pixel*2+1
        self.boundary_kernel = np.ones((boundary_width, boundary_width))

    def add(self, predict, label):
        predict = predict.numpy()
        label = label.numpy()

        self.predict.append(predict)
        self.label.append(label)
        self.boundary_pred.append(self.mask_to_boundary(predict))
        self.boundary_label.append(self.mask_to_boundary(label))
        # print(predict.shape, self.boundary_pred[-1].shape)
        # print(label.shape, self.boundary_label[-1].shape)
    
    def mask_to_boundary(self, mask):
        boundary = mask.astype(np.uint8)
        boundary = np.stack([cv2.erode(m, self.boundary_kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1) for m in boundary])
        return mask-boundary

    def f1(self, e = 1):
        temp_predict = np.array(self.predict).flatten()
        temp_GT = np.array(self.label).flatten()
        tp = np.sum((temp_predict == 1) * (temp_GT == 1))
        fp = np.sum((temp_predict == 1) * (temp_GT == 0))
        fn = np.sum((temp_predict == 0) * (temp_GT == 1))
        precision = tp / (tp+fp+e)
        recall = tp / (tp+fn+e)
        return 2*precision*recall/(precision+recall)

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
    
    def biou(self, e=1):
        temp_predict = np.array(self.predict).flatten()
        temp_GT = np.array(self.label).flatten()
        temp_mask_pred = np.array(self.boundary_pred).flatten()
        temp_mask_label = np.array(self.boundary_label).flatten()
        
        biou = self.compute_iou(temp_predict*temp_mask_pred, temp_GT*temp_mask_label, e)
        iou = self.compute_iou(temp_predict, temp_GT, e)
        return iou, biou
    
    def compute_all(self, e = 1):
        ''' return a dict of losses '''
        iou, biou = self.biou(e)
        f1 = self.f1(e)
        return {
            'IoU': iou, 
            'BIoU': biou,
            'F1': f1
            }

    def clear(self):
        self.predict.clear()
        self.label.clear()
        self.boundary_pred.clear()
        self.boundary_label.clear()
    
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