import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
import torch.nn as nn
import argparse
import torch
from PIL import Image, ImageOps
from torch.utils import data
import time
import random
import segmentation_models_pytorch as smp
from torchvision import transforms as T
import torch
import torch.nn as nn
from torch.autograd import Variable
import imageio
import copy
import warnings
import matplotlib.pyplot as plt

from network.UnetLSTM import New_DeepLabV3Plus_LSTM
from network.models import *
from train_src.dataloader import get_continuous_loader
from all_model import WHICH_MODEL
from config import config_parser_test

# Batch x NumChannels x Height x Width
# UNET --> BatchSize x 1 (3?) x 240 x 240
# BDCLSTM --> BatchSize x 64 x 240 x240

def postprocess_img(o_img, final_mask_exist, continue_list):
    int8_o_img = np.array(o_img, dtype=np.uint8)
    if np.sum(int8_o_img) < 500 or final_mask_exist == 0 or continue_list == 0:
        return np.zeros((o_img.shape[0],o_img.shape[1]), dtype = np.uint8)
    else:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(int8_o_img, connectivity=8)
        index_sort = np.argsort(-stats[:,4])
        if index_sort.shape[0] > 2:
            for ll in (index_sort[2:]):
                labels[ labels == ll ] = 0
        if np.sum(labels) < 500:
            return np.zeros((o_img.shape[0],o_img.shape[1]), dtype = np.uint8)
        else:
            return np.array(labels, dtype=np.uint8)

def Check_continue(continue_list, postprocess_continue_list, bound_list, distance):
    start = 0
    end = 0
    check_start = False
    check_only_list = False 
    for i in range(len(continue_list)):
        if check_only_list == True:
            postprocess_continue_list[i] = 0
        if continue_list[i] == 1 and check_start == False and i < len(continue_list)-1:
            start = i
            check_start = True
        elif continue_list[i] == 1 and check_start == True and i < len(continue_list)-1 and i not in bound_list:
            end = i
        elif continue_list[i] == 0 and check_start == True:
            temp = (end+1) - start
            if temp < 0:
                postprocess_continue_list[start: start+1] = [0]
            if temp <= distance:
                postprocess_continue_list[start: end+1] = [0] * temp
            if temp > distance:
                check_only_list = True
            check_start = False
        elif continue_list[i] == 1 and i in  bound_list:
            end = i
            temp = (end+1) - start
            if temp < 0:
                postprocess_continue_list[end: end+1] = [0]
            if temp <= distance:
                postprocess_continue_list[start: end+1] = [0] * temp
            check_start = False
            check_only_list = False
        elif i in bound_list:
            check_only_list = False
    return postprocess_continue_list

def Cal_mask_center(mask_img):
    middle_list = {}
    for key in mask_img:
        middle_list[key] = []
        for img_index in range(len(mask_img[key])):
            img = mask_img[key][img_index]
            if np.sum(img) != 0:
                mean_x = np.mean(img.nonzero()[0])
                mean_y = np.mean(img.nonzero()[1])
            else:
                mean_x = 0
                mean_y = 0
            middle_list[key].append([mean_x, mean_y])
    return middle_list

def Cal_Local_Global_mean(middle_list, interval_num = 5):
    mean_list = {}
    global_mean_list = {}
    for key in middle_list:
        temp_total = [0] * interval_num
        temp_x = [0] * interval_num
        temp_y = [0] * interval_num
        temp_global_x = 0
        temp_global_y = 0
        temp_global_total = 0
        len_check_list = []
        for i in range(1,interval_num+1):
            len_check_list.append([(i-1), (i-1)*len(middle_list[key])/interval_num, i*len(middle_list[key])/interval_num ])
        for i, (x,y) in enumerate(middle_list[key]):
            if x!=0 and y != 0:
                temp_global_x += x
                temp_global_y += y
                temp_global_total += 1
                for j in range(len(len_check_list)):
                    if i >= len_check_list[j][1] and i< len_check_list[j][2]:
                        temp_x[len_check_list[j][0]] += x
                        temp_y[len_check_list[j][0]] += y
                        temp_total[len_check_list[j][0]] += 1
        if temp_global_total == 0:
            temp_global_total += 1
        for check_temp in range(len(temp_total)):
            if temp_total[check_temp] == 0:
                temp_total[check_temp] +=1
        temp_list = []
        for temp in range(len(temp_total)):
            temp_list.append([ temp_x[temp]/temp_total[temp], temp_y[temp]/temp_total[temp]])
        global_mean_list[key] = [temp_global_x/ temp_global_total, temp_global_y/ temp_global_total]
        mean_list[key] = temp_list
    return mean_list, global_mean_list

def Final_postprocess(middle_list, mean_list, global_mean_list, interval_num , distance ):
    final_mask_exist = []
    for key in middle_list:
        len_check_list = []
        for i in range(1,interval_num+1):
            len_check_list.append([(i-1), (i-1)*len(middle_list[key])/interval_num, i*len(middle_list[key])/interval_num ])
        for i, (x, y) in enumerate(middle_list[key]):
            if x == 0 and y == 0:
                final_mask_exist.append(0)
            else:
                for j in range(len(len_check_list)):
                    if i >= len_check_list[j][1] and i< len_check_list[j][2]:
                        abs_x = min(abs(x-global_mean_list[key][0]),abs(x - mean_list[key][len_check_list[j][0]][0]))
                        abs_y = min(abs(y-global_mean_list[key][1]), abs(y - mean_list[key][len_check_list[j][0]][1]))
                        if abs_x >= distance or abs_y >= distance:
                            final_mask_exist.append(0)
                        else:
                            final_mask_exist.append(1)
    return final_mask_exist


def LISTDIR(path):
    d_list = [f for f in os.listdir(path) if not f.startswith('.')]
    # for f in os.listdir(path):
    #     if not f.startswith('.'):
    #         d_list.append(f)
    d_list.sort()
    return d_list

def read_dir_path(path, dir_str):
    list_dir = []
    for num_files in LISTDIR(path):
        full_path = os.path.join(path, num_files)
        for dir_files in LISTDIR(full_path):
            dir_path = os.path.join(full_path, dir_files, dir_str)
            for files in LISTDIR(dir_path):
                list_dir.append(os.path.join(dir_path, files))
    return list_dir

def test_wo_postprocess(config, test_loader, net):
    if not os.path.isdir(config.output_path):
        print("os.makedirs "+ config.output_path)
        os.makedirs(config.output_path)
    OUTPUT_IMG(config, test_loader, net, False)
    MERGE_VIDEO(config)

def test_w_postprocess(config, test_loader, net):
    net.eval()
    if not os.path.isdir(config.output_path):
        os.makedirs(config.output_path)
    with torch.no_grad():
        Sigmoid_func = nn.Sigmoid()
        threshold = 0.2
        temp_mask_exist, temp_continue_list = [1] * len(test_loader), [1] * len(test_loader)
        mask_img, continue_list, bound_list = {}, [], []
        last_signal, start, end, last_film_name = 0, 0, -1, ""
        for i, (crop_image ,file_name, image) in enumerate(tqdm(test_loader)):
            if config.continuous == 0:
                image = image.cuda()
                output = net(image)
            elif config.continuous == 1:
                pn_frame = image[:,1:,:,:,:]
                frame = image[:,:1,:,:,:]
                temporal_mask, output = net(frame, pn_frame)
                output = output.squeeze(dim = 1)
            output = Sigmoid_func(output)
            SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy()
            SR = postprocess_img(SR, temp_mask_exist[i], temp_continue_list[1])
            if np.sum(SR != 0) == 0:
                continue_list.append(0)
            else:
                continue_list.append(1)
            dict_path = file_name[0].split("/")[-3]
            if dict_path not in mask_img:
                if i != 0:
                    bound_list.append(i-1)
                mask_img[dict_path] = []
                mask_img[dict_path].append(SR)
            else:
                mask_img[dict_path].append(SR)
        bound_list.append(i)
        
        postprocess_continue_list = copy.deepcopy(continue_list)
        postprocess_continue_list = Check_continue(continue_list, postprocess_continue_list, bound_list, distance = 30)
        middle_list = Cal_mask_center(mask_img)
        mean_list, global_mean_list = Cal_Local_Global_mean(middle_list, config.interval_num)
        final_mask_exist = Final_postprocess(middle_list, mean_list, global_mean_list, config.interval_num, config.distance)
        OUTPUT_IMG(config, test_loader, net, True, final_mask_exist, postprocess_continue_list)
        MERGE_VIDEO(config)

def frame2video(path, video_path=None):
    if video_path is None:
        video_path = (path[:-6])
    videoWriter = cv2.VideoWriter(os.path.join(video_path,"merge_video.mp4"), cv2.VideoWriter_fourcc(*'MP4V'), 12.0, (832, 352))
    for frame_files in  LISTDIR(path):
        if frame_files[-3:] == "jpg":
            full_path = os.path.join(path, frame_files)
            frame = cv2.imread(full_path)
            videoWriter.write(frame)
    videoWriter.release()

def film_frame2video(path, video_path=None):
    if video_path is None:
        video_path = (path[:-7])
    videoWriter = cv2.VideoWriter(os.path.join(video_path,"film_video.mp4"), cv2.VideoWriter_fourcc(*'MP4V'), 12.0, (416, 352))
    for frame_files in  LISTDIR(path):
        if frame_files[-3:] == "jpg":
            full_path = os.path.join(path, frame_files)
            frame = cv2.imread(full_path)
            videoWriter.write(frame)
    videoWriter.release()

def OUTPUT_IMG(config, test_loader, net, postprocess = False, final_mask_exist = [], postprocess_continue_list = []):
    if postprocess == False:
        print("No postprocessing!!")
    else:
        print("Has postprocessing!!")
    Sigmoid_func = nn.Sigmoid()
    threshold = 0.5
    with torch.no_grad():
        net.eval()
        tStart = time.time()
        for i, (crop_image ,file_name, image) in enumerate(tqdm(test_loader)):
            pn_frame = image[:,1:,:,:,:]
            frame = image[:,:1,:,:,:]
            
            temporal_mask, output = net(frame, pn_frame)
            output = output.squeeze(dim = 1)
            temporal_mask = Sigmoid_func(temporal_mask)

            temp = [config.output_path] + file_name[0].split("/")[-4:-2]
            write_path = "/".join(temp)
            img_name = file_name[0].split("/")[-1]
            if not os.path.isdir(write_path+"/original"):
                os.makedirs(write_path+"/original")
            if not os.path.isdir(write_path+"/forfilm"):
                os.makedirs(write_path+"/forfilm")
            if not os.path.isdir(write_path+"/merge"):
                os.makedirs(write_path+"/merge")
            if not os.path.isdir(write_path+"/vol_mask"):
                os.makedirs(write_path+"/vol_mask")
            if not os.path.isdir(write_path+"/prob_heat"):
                os.makedirs(write_path+"/prob_heat")

            output = Sigmoid_func(output)
            crop_image = crop_image.squeeze().data.numpy()
            origin_crop_image = crop_image.copy()
            # prob_heatmap = output.squeeze().cpu().data.numpy()
            SR = torch.where(output > threshold, 1, 0).squeeze().cpu().data.numpy()
            if postprocess == True:
                SR = postprocess_img(SR, final_mask_exist[i], postprocess_continue_list[i])
                SR = np.where(SR > 0.5, 1, 0)
            heatmap = np.uint8(110 * SR)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
            # prob_heatmap = np.uint8(255*prob_heatmap)
            # prob_heatmap = cv2.applyColorMap(prob_heatmap, cv2.COLORMAP_JET)
            
            heat_img = heatmap*0.6+origin_crop_image
            # prob_img = prob_heatmap*0.6+origin_crop_image
            merge_img = np.hstack([origin_crop_image, heat_img])
            cv2.imwrite(os.path.join(write_path+"/merge", img_name), merge_img)
            imageio.imwrite(os.path.join(write_path+"/original", img_name), origin_crop_image)
            cv2.imwrite(os.path.join(write_path+"/forfilm", img_name), heat_img)
            # cv2.imwrite(os.path.join(write_path+"/prob_heat", img_name), prob_img)
            cv2.imwrite(os.path.join(write_path+"/vol_mask", img_name), SR*255)
        tEnd = time.time()
        print("Cost time(seconds)= "+str(tEnd-tStart))

def MERGE_VIDEO(config, copy_height=True):
    for dir_files in (LISTDIR(config.output_path)):
        full_path = os.path.join(config.output_path, dir_files)
        o_full_path = os.path.join(config.output_img_path, dir_files)
        if os.path.isdir(full_path):
            for num_files in tqdm(LISTDIR(full_path)):
                full_path_2 = os.path.join(full_path, num_files+"/merge")
                full_path_3 = os.path.join(full_path, num_files+"/forfilm")
                height_path = os.path.join(o_full_path, num_files, "height.txt")
                s_height_path = os.path.join(full_path, num_files)
                if copy_height: os.system("cp "+height_path+" "+s_height_path)
                frame2video(full_path_2)
                # full_path_4 = os.path.join(full_path, num_files+"/prob_heat")
                # frame2video(full_path_4, video_path=full_path_4)
                film_frame2video(full_path_3)
                if config.keep_image == 0:
                    full_path_3 = os.path.join(full_path, num_files+"/original")
                    os.system("rm -r "+full_path_3)
                    full_path_3 = os.path.join(full_path, num_files+"/forfilm")
                    os.system("rm -r "+full_path_3)


def vol_cal(frame_file_path):
    pixel_area = (1 / 85)**2  # h = 10/85(mm)
    for num_files in LISTDIR(frame_file_path):
        full_path = os.path.join(frame_file_path, num_files)
        for date_file in LISTDIR(full_path):
            full_path_2 = os.path.join(full_path, date_file)
            h_path = "/".join(['output_frame']+full_path_2.split("/")[1:])
            height_file = os.path.join(h_path, "height.txt")
            f = open(height_file, "r")
            height = float(f.read())/10.0
            f.close()
            mask_path = os.path.join(full_path_2, "vol_mask")
            mask_list = []
            for mask_file in LISTDIR(mask_path):
                full_path_3 = os.path.join(mask_path, mask_file)
                mask = cv2.imread(full_path_3, cv2.IMREAD_GRAYSCALE)
                mask_list.append(mask)
            left_area = 0.0
            right_area = 0.0
            area = []
            index = []
            for i in range(0, len(mask_list)-1):
                left_area += np.count_nonzero(mask_list[i] != 0.0)
                temp_area = np.count_nonzero(mask_list[i] != 0.0) * pixel_area / len(mask_list)
                area.append(temp_area)
                index.append(i)
            temp_area = np.count_nonzero(mask_list[len(mask_list)-1] != 0.0) * pixel_area / len(mask_list)
            area.append(temp_area)
            index.append(len(mask_list)-1)
            for i in range(1, len(mask_list)):
                right_area += np.count_nonzero(mask_list[i] != 0.0)    
            ll = (left_area * pixel_area)
            rr = (right_area * pixel_area)
            result = (ll+rr)*height/(2* np.sum(np.array(area) != 0))
            plt.title("Volume: "+str(round(result,4))+" ml")
            plt.xlabel("frame index")
            plt.ylabel("area(cm^2)")
            plt.bar(index, area, width = 1.5)
            save_path = os.path.join(full_path_2, "Volume.jpg")
            plt.savefig(save_path)
            plt.close()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    print(os.getcwd())
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    seed = 1029
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    config = config_parser_test()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for num_file in LISTDIR(config.video_path):
        # Input folder
        full_path = os.path.join(config.video_path, num_file)

        for video_file in LISTDIR(full_path):
            # Patient ID (PXXX) folder
            full_path_2 = os.path.join(full_path, video_file)

            for video_file in LISTDIR(full_path_2):
                video_path = os.path.join(full_path_2, video_file)
                # Video info folder

                if video_path.split(".")[-1] in ['avi', 'mp4']:
                    vidcap = cv2.VideoCapture(video_path)
                    success,image = vidcap.read()
                    count = 0
                    success = True
                    dir_write_path = os.path.join( config.output_img_path,"/".join(video_path.split("/")[-3:-2]), "_".join(video_path.split("/")[-2:]).split(".")[0])
                    write_path = os.path.join(dir_write_path, "original")
                    if not os.path.isdir(write_path):
                        os.makedirs(write_path)
                    height_path = os.path.join(full_path_2, "height.txt")
                    os.system("cp "+ height_path+" "+dir_write_path)
                    while success:
                        image = cv2.resize(image, (720, 540), cv2.INTER_CUBIC)
                        if count<10:
                            cv2.imwrite(write_path+"/frame00%d.jpg" % count, image) 
                        elif count < 100 and count > 9:
                            cv2.imwrite(write_path+"/frame0%d.jpg" % count, image)
                        elif count > 99: 
                            cv2.imwrite(write_path+"/frame%d.jpg" % count, image)
                        count += 1 
                        success,image = vidcap.read()
    print("video to frame finised!")

    o_files = read_dir_path(config.output_img_path, "original")
    for files in o_files:
        img = Image.open(files).convert('RGB')
        if img.size != (720, 540):
            img = img.resize((720, 540))
        img = img.crop((148, 72, 571, 424))
        img = img.resize((416, 352))
        img.save(files)
    print("image croped finished!")

    with torch.no_grad():
        frame_continue_num = config.continue_num
        test_loader, continue_num = get_continuous_loader(image_path = config.output_img_path,
                                    batch_size = 1,
                                    mode = 'test',
                                    augmentation_prob = 0.,
                                    shffule_yn = False,
                                    continue_num = frame_continue_num)
        
        # net = New_DeepLabV3Plus_LSTM(1, (3, 8), len(frame_continue_num), "resnet34")
        net, model_name = WHICH_MODEL(config, frame_continue_num)
        print(model_name)
        # net = DeepLabV3Plus_SA(1, (3, 8), len(frame_continue_num))
        # model_name = "New_DeepLabV3Plus_LSTM"+"_"+"resnet34"

        # net.load_state_dict(torch.load(config.model_path, map_location='cpu'))        
        # print("pretrain model loaded!")

        net = net.to(device)
        test_w_postprocess(config, test_loader, net)
        print("image sequences prediction finished!")
        vol_result = vol_cal(config.output_path)
        print("Volume Estimation finished!")
    