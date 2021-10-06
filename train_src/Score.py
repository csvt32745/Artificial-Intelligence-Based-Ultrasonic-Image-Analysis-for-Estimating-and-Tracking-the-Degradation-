import numpy as np
from sklearn.metrics import fbeta_score
import os
import cv2
from tqdm import tqdm
class Scorer():
    def __init__(self, config):
        self.predict = []
        self.label = []
        self.t = config.threshold
    def add(self, predict, label):
        self.predict += predict.flatten().tolist()
        self.label += label.flatten().tolist()
    def f1(self, e = 1):
        temp_predict = np.array(self.predict)
        temp_GT = np.array(self.label)
        tp = np.sum((temp_predict == 1) * (temp_GT == 1))
        fp = np.sum((temp_predict == 1) * (temp_GT == 0))
        fn = np.sum((temp_predict == 0) * (temp_GT == 1))
        precision = tp / (tp+fp+e)
        recall = tp / (tp+fn+e)
        return 2*precision*recall/(precision+recall)
    def iou(self, e = 1):
        temp_predict = np.array(self.predict)
        temp_GT = np.array(self.label)
        tp_fp = np.sum(temp_predict == 1)
        tp_fn = np.sum(temp_GT == 1)
        tp = np.sum((temp_predict == 1) * (temp_GT == 1))
        iou = tp / (tp_fp + tp_fn - tp+e)
        return iou
    def clear(self):
        self.predict.clear()
        self.label.clear()
    
class Losser():
    def __init__(self):
        self.loss = []
    def add(self, loss_item):
        self.loss.append(loss_item)
    def mean(self):
        return sum(self.loss) / len(self.loss)
        