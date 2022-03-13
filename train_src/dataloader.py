import os
import numpy as np
import torch
import random
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import ipdb
from torch.nn.utils.rnn import pad_sequence
def label_mask(m_array):
    new_mrray = np.zeros((m_array.shape[0],m_array.shape[1]))
    new_mrray[m_array >= 128 ] = 1
    return new_mrray

def preprocess_img(image):
    Transform = []
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)
    image = Transform(image)
    Norm_ = T.Normalize((0.486, 0.456, 0.406), (0.229, 0.224, 0.225))
    image = Norm_(image)
    return image

def denormalize_img_rgb(image):
    # shape: np.ndarray | torch.Tensor (..., 3), [0, 1]
    dim = len(image.shape)
    mean = [0.486, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if isinstance(image, np.ndarray):
        mean = np.array(mean)
        std = np.array(std)
    else:
        mean = torch.FloatTensor(mean)
        std = torch.FloatTensor(std)
    shape = (*([1]*(dim-1)), 3) # len((1, 1, ..., 3)) = dim
    mean = mean.reshape(shape)
    std = std.reshape(shape)
    return image*std + mean

def preprocess_mask(mask):
    mask = np.array(mask)
    mask = label_mask(mask)
    return mask
def test_preprocess_img(image):
    crop_origin_image = image
    Transform = []
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)
    image = Transform(image)
    Norm_ = T.Normalize((0.486, 0.456, 0.406), (0.229, 0.224, 0.225))
    image = Norm_(image)
    return crop_origin_image, image
def read_img_continuous(continuous_frame_num, temp_img_list ,img_dir_file, index):
    list_num = []
    frame_num = int(temp_img_list[index].split("/")[-1].split(".")[0][-3:])
    for check_frame in continuous_frame_num:
        if frame_num + check_frame < 0:
            file_path = img_dir_file+"/frame" + "%03d" % 0 + ".jpg"
        elif frame_num + check_frame > len(temp_img_list) - 1:
            file_path = img_dir_file+"/frame"+ "%03d" % (len(temp_img_list) - 1) + ".jpg"
        else:
            file_path = img_dir_file+ "/frame"+ "%03d" % (frame_num + check_frame)+ ".jpg"
        if not os.path.isfile(file_path):
            file_path = img_dir_file+"/frame" + "%03d"% frame_num+".jpg"
        list_num.append(file_path)
    return frame_num, list_num
def read_mask_continuous(continuous_frame_num, temp_mask_list, mask_dir_file, index):
    list_num = []
    frame_num = int(temp_mask_list[index].split("/")[-1].split(".")[0][5:8])
    for check_frame in continuous_frame_num:
        if frame_num + check_frame < 0:
            mask_path = mask_dir_file+"/frame" + "%03d" % 0 + "_out.jpg"
        elif frame_num + check_frame > len(temp_mask_list) - 1:
            mask_path = mask_dir_file+"/frame"+ "%03d" % (len(temp_mask_list) - 1) + "_out.jpg"
        else:
            mask_path = mask_dir_file+ "/frame"+ "%03d" % (frame_num + check_frame)+ "_out.jpg"
        if not os.path.isfile(mask_path):
            mask_path = mask_dir_file+ "/frame" + "%03d"% frame_num+"_out.jpg"
        list_num.append(mask_path)
    return frame_num, list_num
class ImageFolder(data.Dataset):
    def __init__(self, root, prob, mode = 'train'):
        self.root = root
        if mode == "train" or mode == "valid":
            self.image_paths = []
            self.mask_paths = []
            for num_file in os.listdir(self.root):
                full_path = os.path.join(self.root, num_file)
                for dir_file in os.listdir(full_path):
                    temp_img_list = []
                    temp_mask_list = []
                    full_path_2 = os.path.join(full_path, dir_file)
                    for original_file in os.listdir(os.path.join(full_path_2, "original")):
                        full_path_3 = os.path.join(os.path.join(full_path_2, "original", original_file))
                        temp_img_list.append(full_path_3)
                    for original_file in os.listdir(os.path.join(full_path_2, "mask")):
                        full_path_3 = os.path.join(os.path.join(full_path_2, "mask", original_file))
                        temp_mask_list.append(full_path_3)
                    temp_img_list.sort(key = lambda x : (x.split("/")[-1].split("_")[0]))
                    temp_mask_list.sort(key = lambda x : (x.split("/")[-1].split("_")[0]))
                    self.image_paths += temp_img_list
                    self.mask_paths += temp_mask_list
            new_img_path = []
            for img in self.image_paths:
                img = img.replace("original", "mask")
                new_img_path.append(img.replace(".jpg", "_out.jpg"))
            if new_img_path == self.mask_paths:
                print("Image and Masks are correct")
            else:
                for i in range(len(self.image_paths)):
                    img = new_img_path[i]
                    mask = self.mask_paths[i]
                    if img != mask:
                        print(img,mask,i)
        if mode == "test":
            self.image_paths = []
            for num_file in os.listdir(self.root):
                full_path = os.path.join(self.root, num_file)
                for dir_file in os.listdir(full_path):
                    temp_img_list = []
                    full_path_2 = os.path.join(full_path, dir_file)
                    for original_file in os.listdir(os.path.join(full_path_2, "original")):
                        full_path_3 = os.path.join(os.path.join(full_path_2, "original", original_file))
                        temp_img_list.append(full_path_3)
                    temp_img_list.sort(key = lambda x : (x.split("/")[-1].split("_")[0]))
                    self.image_paths += temp_img_list
        self.mode = mode
        self.augmentation_prob = prob
        self.RotationDegree = [0,90,180,270]
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))
    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "valid":
            image_path = self.image_paths[index]
            mask_path = self.mask_paths[index]
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert("L")
            image = preprocess_img(image)
            mask = preprocess_mask(mask)
            mask = torch.tensor(mask, dtype=torch.long) 
            if self.augmentation_prob > np.random.rand():
                transform = T.Compose([
                T.RandomHorizontalFlip(p = 1.0),
                T.RandomVerticalFlip(p = 1.0),
                ])
                image = transform(image)
                mask = transform(mask)
            return image, mask    
        if self.mode == "test":
            file_name = self.image_paths[index]
            image_path = self.image_paths[index]
            image = Image.open(image_path).convert('RGB')
            crop_origin_image, image = test_preprocess_img(image)
            return  np.array(crop_origin_image), file_name, image
            
    def __len__(self):
        return len(self.image_paths)

def get_loader(image_path, batch_size, mode, augmentation_prob, shffule_yn = False):
    dataset = ImageFolder(root = image_path, prob = augmentation_prob,mode = mode)
    data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=shffule_yn,
                                  drop_last=True)
    return data_loader
class Continuos_Image(data.Dataset):
    def __init__(self, root, prob, mode = 'train', continuous_frame_num = [1, 2, 3, 4, 5, 6, 7, 8], data_ratio=1):
        self.root = root
        self.mode = mode
        self.augmentation_prob = prob
        self.continuous_frame_num = continuous_frame_num
        if mode == "train" or mode == "valid":
            self.image_paths = {}
            self.mask_paths = {}
            for num_file in os.listdir(self.root):
                full_path = os.path.join(self.root, num_file)
                for dir_file in os.listdir(full_path):
                    temp_img_list = []
                    temp_mask_list = []
                    full_path_2 = os.path.join(full_path, dir_file)
                    for original_file in os.listdir(os.path.join(full_path_2, "original")):
                        full_path_3 = os.path.join(os.path.join(full_path_2, "original", original_file))
                        temp_img_list.append(full_path_3)
                    for original_file in os.listdir(os.path.join(full_path_2, "mask")):
                        full_path_3 = os.path.join(os.path.join(full_path_2, "mask", original_file))
                        temp_mask_list.append(full_path_3)
                    temp_img_list.sort(key = lambda x : (x.split("/")[-1].split("_")[0]))
                    temp_mask_list.sort(key = lambda x : (x.split("/")[-1].split("_")[0]))
                    img_dir_file = "/".join(temp_img_list[0].split("/")[:-1])
                    mask_dir_file = "/".join(temp_mask_list[0].split("/")[:-1])
                    total_img_num = []
                    total_mask_num = []
                    for i in range(len(temp_img_list)):
                        frame_num, list_num = read_img_continuous(self.continuous_frame_num, temp_img_list, img_dir_file, i)
                        order_num = [img_dir_file+"/frame" + "%03d"% frame_num+".jpg"] + list_num
                        total_img_num.append(order_num)
                    for i in range(len(temp_mask_list)):
                        frame_num, list_num = read_mask_continuous(self.continuous_frame_num, temp_mask_list, mask_dir_file, i)
                        order_num = [mask_dir_file+ "/frame" + "%03d"% frame_num+"_out.jpg"] + list_num
                        total_mask_num.append(order_num)
                    self.image_paths[img_dir_file] = total_img_num
                    self.mask_paths[mask_dir_file] = total_mask_num
            temp_list = [*self.image_paths.values()]
            temp_mask_list = [*self.mask_paths.values()]
            if data_ratio < 1:
                data_ratio = int(np.ceil(len(temp_list)*data_ratio))
                # z = list(zip(temp_list, temp_mask_list))
                # np.random.shuffle(z)
                # temp_list, temp_mask_list = zip(*z)
            else:
                data_ratio = len(temp_list)
            self.image_paths_list = [val for sublist in temp_list[:data_ratio] for val in sublist]
            self.mask_paths_list = [val for sublist in temp_mask_list[:data_ratio] for val in sublist]
            print(self.image_paths_list[100], self.mask_paths_list[100])
        if mode == "test":
            self.image_paths = {}
            for num_file in os.listdir(self.root):
                full_path = os.path.join(self.root, num_file)
                for dir_file in os.listdir(full_path):
                    temp_img_list = []
                    full_path_2 = os.path.join(full_path, dir_file)
                    for original_file in os.listdir(os.path.join(full_path_2, "original")):
                        full_path_3 = os.path.join(os.path.join(full_path_2, "original", original_file))
                        temp_img_list.append(full_path_3)
                    temp_img_list.sort(key = lambda x : (x.split("/")[-1].split("_")[0]))
                    img_dir_file = "/".join(temp_img_list[0].split("/")[:-1])
                    total_img_num = []
                    for i in range(len(temp_img_list)):
                        frame_num, list_num = read_img_continuous(self.continuous_frame_num, temp_img_list, img_dir_file, i) 
                        order_num = [img_dir_file+ "/frame" + "%03d"% frame_num+".jpg"] + list_num
                        total_img_num.append(order_num)
                    self.image_paths[img_dir_file] = total_img_num
            temp_list = [*self.image_paths.values()]
            self.image_paths_list = [val for sublist in temp_list for val in sublist]
        print("image count in {} path :{}".format(self.mode,len(self.image_paths_list)))
    def __getitem__(self, index):
        dist_x = 416
        dist_y = 352
        if self.mode == "train" or self.mode == "valid":
            image_list = self.image_paths_list[index]
            mask_list = self.mask_paths_list[index]
            image = torch.tensor([]).cuda()
            mask = []
            for image_path in image_list:
                i_image = Image.open(image_path).convert('RGB')
                image = torch.cat((image, preprocess_img(i_image).cuda()), dim = 0)
            image = image.view(-1, 3, dist_y, dist_x)
            for mask_path in mask_list:
                i_mask = Image.open(mask_path).convert("L")
                mask.append(preprocess_mask(i_mask))
            mask = np.array(mask)
            return image_list, image, mask
        if self.mode == "test":
            image_list = self.image_paths_list[index]
            image = torch.tensor([]).cuda()
            for i, image_path in enumerate(image_list):
                i_image = Image.open(image_path).convert('RGB')
                image = torch.cat((image, test_preprocess_img(i_image)[1].cuda()), dim = 0)
                if i == 0:
                    o_image = np.array(test_preprocess_img(i_image)[0])
            image = image.view(-1, 3, dist_y, dist_x)
            return o_image, image_list[0], image
    def __len__(self):
        return len(self.image_paths_list)
    def its_continue_num(self):
        return self.continuous_frame_num
def get_continuous_loader(image_path, batch_size, mode, augmentation_prob, shffule_yn = False, continue_num = [1, 2, 3, 4, 5, 6, 7, 8], smaller_dataset=1):
    dataset = Continuos_Image(root = image_path, prob = augmentation_prob,mode = mode, continuous_frame_num = continue_num, data_ratio = smaller_dataset)
    data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=shffule_yn,
                                  drop_last=True )
    return data_loader, dataset.its_continue_num()
