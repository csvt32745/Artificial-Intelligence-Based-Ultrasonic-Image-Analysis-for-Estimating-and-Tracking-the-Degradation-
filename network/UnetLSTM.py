from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
import warnings
import segmentation_models_pytorch as smp
from network.CLSTM import BDCLSTM
from network.CLSTM import New_BDCLSTM, Temp_New_BDCLSTM

from typing import Tuple

class Unet_LSTM(nn.Module):
    def __init__(self, num_classes, continue_num = 8, backbone = "resnet34"):
        super().__init__()
        self.unet1 = smp.Unet(
            encoder_name=backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        self.len = continue_num
        self.lstm = BDCLSTM(input_channels = 1, hidden_channels=[8])
    def forward(self, input, other_frame):
        predict_pre = []
        predict_next = []
        temporal_mask = torch.tensor([]).cuda()
        for i in range(int(self.len / 2)):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_pre.append(temp)
        for i in range(int(self.len / 2+1), self.len):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_next.append(temp)
        predict_now = self.unet1(other_frame[:,self.len // 2:self.len // 2+1,:,:,:].squeeze(dim = 1))
        final_predict = self.lstm(predict_pre, predict_now, predict_next)
        for p_p in (predict_pre):
            temporal_mask = torch.cat((temporal_mask, p_p), dim = 1)
        temporal_mask = torch.cat((temporal_mask, predict_now), dim = 1)
        for p_n in (predict_next):
            temporal_mask = torch.cat((temporal_mask, p_n), dim = 1)
        return temporal_mask, final_predict

class UnetPlusPlus_LSTM(nn.Module):
    def __init__(self, num_classes, continue_num = 8, backbone = "resnet34"):
        super().__init__()
        self.unet1 = smp.UnetPlusPlus(
            encoder_name=backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        self.len = continue_num
        self.lstm = BDCLSTM(input_channels = 1, hidden_channels=[8])
    def forward(self, input, other_frame):
        predict_pre = []
        predict_next = []
        temporal_mask = torch.tensor([]).cuda()
        for i in range(int(self.len / 2)):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_pre.append(temp)
        for i in range(int(self.len / 2+1), self.len):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_next.append(temp)
        predict_now = self.unet1(other_frame[:,self.len // 2:self.len // 2+1,:,:,:].squeeze(dim = 1))
        final_predict = self.lstm(predict_pre, predict_now, predict_next)
        for p_p in (predict_pre):
            temporal_mask = torch.cat((temporal_mask, p_p), dim = 1)
        temporal_mask = torch.cat((temporal_mask, predict_now), dim = 1)
        for p_n in (predict_next):
            temporal_mask = torch.cat((temporal_mask, p_n), dim = 1)
        return temporal_mask, final_predict
class PSPNet_LSTM(nn.Module):
    def __init__(self, num_classes, continue_num = 8, backbone = "resnet34"):
        super().__init__()
        self.unet1 = smp.PSPNet(
            encoder_name=backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        self.len = continue_num
        self.lstm = BDCLSTM(input_channels = 1, hidden_channels=[8])
    def forward(self, input, other_frame):
        predict_pre = []
        predict_next = []
        temporal_mask = torch.tensor([]).cuda()
        for i in range(int(self.len / 2)):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_pre.append(temp)
        for i in range(int(self.len / 2+1), self.len):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_next.append(temp)
        predict_now = self.unet1(other_frame[:,self.len // 2:self.len // 2+1,:,:,:].squeeze(dim = 1))
        final_predict = self.lstm(predict_pre, predict_now, predict_next)
        for p_p in (predict_pre):
            temporal_mask = torch.cat((temporal_mask, p_p), dim = 1)
        temporal_mask = torch.cat((temporal_mask, predict_now), dim = 1)
        for p_n in (predict_next):
            temporal_mask = torch.cat((temporal_mask, p_n), dim = 1)
        return temporal_mask, final_predict

class DeepLabV3Plus_LSTM(nn.Module):
    def __init__(self, num_classes, continue_num = 8, backbone = "resnet34"):
        super().__init__()
        self.unet1 = smp.DeepLabV3Plus(
            encoder_name=backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        self.len = continue_num
        self.lstm = BDCLSTM(input_channels = 1, hidden_channels=[8])
    def forward(self, input, other_frame):
        predict_pre = []
        predict_next = []
        temporal_mask = torch.tensor([]).cuda()
        for i in range(int(self.len / 2)):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_pre.append(temp)
        for i in range(int(self.len / 2+1), self.len):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_next.append(temp)
        predict_now = self.unet1(other_frame[:,self.len // 2:self.len // 2+1,:,:,:].squeeze(dim = 1))
        final_predict = self.lstm(predict_pre, predict_now, predict_next)
        for p_p in (predict_pre):
            temporal_mask = torch.cat((temporal_mask, p_p), dim = 1)
        temporal_mask = torch.cat((temporal_mask, predict_now), dim = 1)
        for p_n in (predict_next):
            temporal_mask = torch.cat((temporal_mask, p_n), dim = 1)
        return temporal_mask, final_predict

class DeepLabV3_LSTM(nn.Module):
    def __init__(self, num_classes, continue_num = 8, backbone = "resnet34"):
        super().__init__()
        self.unet1 = smp.DeepLabV3(
            encoder_name=backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        self.len = continue_num
        self.lstm = BDCLSTM(input_channels = 1, hidden_channels=[8])
    def forward(self, input, other_frame):
        predict_pre = []
        predict_next = []
        temporal_mask = torch.tensor([]).cuda()
        for i in range(int(self.len / 2)):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_pre.append(temp)
        for i in range(int(self.len / 2+1), self.len):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_next.append(temp)
        predict_now = self.unet1(other_frame[:,self.len // 2:self.len // 2+1,:,:,:].squeeze(dim = 1))
        final_predict = self.lstm(predict_pre, predict_now, predict_next)
        for p_p in (predict_pre):
            temporal_mask = torch.cat((temporal_mask, p_p), dim = 1)
        temporal_mask = torch.cat((temporal_mask, predict_now), dim = 1)
        for p_n in (predict_next):
            temporal_mask = torch.cat((temporal_mask, p_n), dim = 1)
        return temporal_mask, final_predict

class Linknet_LSTM(nn.Module):
    def __init__(self, num_classes, continue_num = 8, backbone = "resnet34"):
        super().__init__()
        self.unet1 = smp.Linknet(
            encoder_name=backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )
        self.len = continue_num
        self.lstm = BDCLSTM(input_channels = 1, hidden_channels=[8])
    def forward(self, input, other_frame):
        predict_pre = []
        predict_next = []
        temporal_mask = torch.tensor([]).cuda()
        for i in range(int(self.len / 2)):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_pre.append(temp)
        for i in range(int(self.len / 2+1), self.len):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_next.append(temp)
        predict_now = self.unet1(other_frame[:,self.len // 2:self.len // 2+1,:,:,:].squeeze(dim = 1))
        final_predict = self.lstm(predict_pre, predict_now, predict_next)
        for p_p in (predict_pre):
            temporal_mask = torch.cat((temporal_mask, p_p), dim = 1)
        temporal_mask = torch.cat((temporal_mask, predict_now), dim = 1)
        for p_n in (predict_next):
            temporal_mask = torch.cat((temporal_mask, p_n), dim = 1)
        return temporal_mask, final_predict

class New_DeepLabV3Plus_LSTM(nn.Module):
    def __init__(self, num_classes,
        hidden_channels: Tuple[int, int] = (3, 8),
        continue_num = 8, backbone = "resnet34"
        ):

        super().__init__()
        self.unet1 = smp.DeepLabV3Plus(
            encoder_name=backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=hidden_channels[0], # model output channels (number of classes in your dataset)
        )
        self.len = continue_num
        self.lstm = New_BDCLSTM(length = continue_num, input_channels = hidden_channels[0], hidden_channels=[hidden_channels[1]])
    def forward(self, input, other_frame):
        temporal_mask = torch.tensor([]).cuda()
        continue_list = []
        for i in range(self.len):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            continue_list.append(temp)
        
        final_predict, temporal_mask = self.lstm(continue_list)
        return temporal_mask, final_predict


class Temp_New_DeepLabV3Plus_LSTM(nn.Module):
    def __init__(self, num_classes, continue_num = 8, backbone = "resnet34"):
        super().__init__()
        self.unet1 = smp.DeepLabV3Plus(
            encoder_name=backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=3,                      # model output channels (number of classes in your dataset)
        )
        self.len = continue_num
        self.lstm = Temp_New_BDCLSTM(input_channels = 3, hidden_channels=[8])
    def forward(self, input, other_frame):
        predict_pre = []
        predict_next = []
        temporal_mask = torch.tensor([]).cuda()
        for i in range(int(self.len / 2)):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_pre.append(temp)
        for i in range(int(self.len / 2+1), self.len):
            temp = self.unet1(other_frame[:,i:i+1,:,:,:].squeeze(dim = 1))
            predict_next.append(temp)
        predict_now = self.unet1(other_frame[:,self.len // 2:self.len // 2+1,:,:,:].squeeze(dim = 1))
        final_predict, temporal_mask = self.lstm(predict_pre, predict_now, predict_next)
        return temporal_mask, final_predict