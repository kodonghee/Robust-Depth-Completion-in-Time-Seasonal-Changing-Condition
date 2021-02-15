import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
import transforms as transforms
import pathlib
import pandas as pd
from PIL import Image
import cv2
import pdb

RGB_IMG_EXTENSIONS = ['.jpg',]
DEP_IMG_EXTENSIONS = ['.png',]
SKY_IMG_EXTENSIONS = ['.png',] # sky

def is_rgb_image_file(filename):
    return any(filename.endswith(extension) for extension in RGB_IMG_EXTENSIONS)

def is_dep_image_file(filename):
    return any(filename.endswith(extension) for extension in DEP_IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth

def get_images_and_labels(rgb_data_root_dir, dep_data_root_dir, sky_data_root_dir):
    rgb_data_root = pathlib.Path(rgb_data_root_dir)
    dep_data_root = pathlib.Path(dep_data_root_dir)
    sky_data_root = pathlib.Path(sky_data_root_dir)
    all_rgb_image_path = [str(path) for path in list(rgb_data_root.glob('*.jpg'))]
    all_dep_image_path = [str(path) for path in list(dep_data_root.glob('*.png'))]
    all_sky_image_path = [str(path) for path in list(sky_data_root.glob('*.png'))]

    return all_rgb_image_path, all_dep_image_path, all_sky_image_path

def get_dataset(rgb_dataset_root_dir, dep_dataset_root_dir, sky_dataset_root_dir):
    all_rgb_image_path, all_dep_image_path, all_sky_image_path = get_images_and_labels(rgb_data_root_dir=rgb_dataset_root_dir, dep_data_root_dir=dep_dataset_root_dir, sky_data_root_dir=sky_dataset_root_dir)
    images = []
    for n, (rgb_path, dep_path, sky_path) in enumerate(zip(all_rgb_image_path, all_dep_image_path, all_sky_image_path)):
        images.append((rgb_path, dep_path, sky_path))

    return images

def loader(path):
    rgb_raw = cv2.imread(path[0], cv2.IMREAD_ANYCOLOR)
    dep_raw = cv2.imread(path[1], cv2.IMREAD_ANYDEPTH)
    sky_raw = cv2.imread(path[2], cv2.IMREAD_ANYCOLOR)
    rgb = np.array(rgb_raw)
    depth = np.array(dep_raw)
    sky = np.array(sky_raw)

    for x in range(sky.shape[0]):
        for y in range(sky.shape[1]):
            if sky[x, y][0] == 255 and sky[x, y][1] == 200 and sky[x, y][2] == 90:
            # if sky[x, y][0] == 128 and sky[x, y][1] == 128 and sky[x, y][2] == 128: # syn data
                sky[x, y] = 1

    sky = np.where(sky != 1, 0, sky)
    img_mask = np.zeros(sky.shape)
    img_mask[sky == 0] = 1
    img_mask[sky == 1] = 0
    img_mask = np.array(img_mask, dtype=np.uint8)
    img_mask_d = img_mask[:, :, 0]
    masked_rgb = rgb * img_mask
    masked_dep = depth * img_mask_d
    # depth = depth/100
    # depth = depth * 255 / depth.max() #former method
    return rgb, depth, masked_rgb, masked_dep

# def rgb2grayscale(rgb):
#     return rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114

to_tensor = transforms.ToTensor()

class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd'] # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, rgb_root, dep_root, sky_root, type, sparsifier=None, modality='rgb', loader=loader):
        classes, class_to_idx = find_classes(rgb_root)
        imgs = get_dataset(rgb_root, dep_root, sky_root)
        assert len(imgs)>0, "Found 0 images in subfolders of: " + rgb_root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.rgb_root = rgb_root
        self.dep_root = dep_root
        self.sky_root = sky_root #sky
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        if type == 'train':
            self.transform = self.train_transform
        elif type == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = loader
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def create_sparse_depth(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def __getraw__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path = self.imgs[index]
        rgb, depth, masked_rgb, masked_depth = self.loader(path)
        return rgb, depth, masked_rgb, masked_depth

    def __getitem__(self, index):
        rgb, depth, masked_rgb, masked_depth = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(masked_rgb, masked_depth)
            # masked_rgb, masked_depth = self.transform(masked_rgb, masked_depth)
        else:
            raise(RuntimeError("transform not defined"))

        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)

        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_np)
        elif self.modality == 'd':
            input_np = self.create_sparse_depth(rgb_np, depth_np)

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)

    # def __get_all_item__(self, index):
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: (input_tensor, depth_tensor, input_np, depth_np)
    #     """
    #     rgb, depth = self.__getraw__(index)
    #     if self.transform is not None:
    #         rgb_np, depth_np = self.transform(rgb, depth)
    #     else:
    #         raise(RuntimeError("transform not defined"))

    #     # color normalization
    #     # rgb_tensor = normalize_rgb(rgb_tensor)
    #     # rgb_np = normalize_np(rgb_np)

    #     if self.modality == 'rgb':
    #         input_np = rgb_np
    #     elif self.modality == 'rgbd':
    #         input_np = self.create_rgbd(rgb_np, depth_np)
    #     elif self.modality == 'd':
    #         input_np = self.create_sparse_depth(rgb_np, depth_np)

    #     input_tensor = to_tensor(input_np)
    #     while input_tensor.dim() < 3:
    #         input_tensor = input_tensor.unsqueeze(0)
    #     depth_tensor = to_tensor(depth_np)
    #     depth_tensor = depth_tensor.unsqueeze(0)

    #     return input_tensor, depth_tensor, input_np, depth_np