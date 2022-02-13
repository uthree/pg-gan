import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import threading
import time
import shutil

from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm
import numpy as np
import multiprocessing

import joblib

class ImageDataset(torch.utils.data.Dataset):
    """Some Information about ImageDataset"""
    def __init__(self, source_dir_pathes=[], chache_dir="./dataset_chache/", size=4, max_len=100000):
        super(ImageDataset, self).__init__()
        self.image_path_list = []
        for dir_path in source_dir_pathes:
            self.image_path_list += [ os.path.join(dir_path, p) for p in  os.listdir(dir_path) ]
        self.chache_dir = chache_dir
        self.image_path_list = self.image_path_list[:max_len]
        self.size = -1
        if not os.path.exists(chache_dir):
            os.mkdir(chache_dir)
    
    def set_size(self, size):
        if self.size == size:
            return
        self.size = size
        
        # initialize directory
        shutil.rmtree(self.chache_dir)
        # resize image and save to chache directory
        if not os.path.exists(self.chache_dir):
            os.mkdir(self.chache_dir)
        
        print("resizing images... to size: {}".format(size))
        def fn(i):
            img_path = self.image_path_list[i]
            img = Image.open(img_path)
            # get height and width
            H, W = img.size
            if H > W:
                W = int(W * size / H)
                H = size
            else:
                H = int(H * size / W)
                W = size
            flag_blur = False
            if img.size[0] > H/2 or img.size[1] > W/2:
                flag_blur = True
            img = img.resize((H, W), Image.NEAREST)
            if flag_blur:
                img = img.filter(ImageFilter.GaussianBlur(1))
            # padding to square
            empty = Image.new("RGB", (size, size), (0, 0, 0))
            # paste image to empty
            empty.paste(img, ((size - H) // 2, (size - W) // 2))
            # save to chache directory
            path = os.path.join(self.chache_dir, str(i) + ".jpg")
            empty.save(path)
            del img
            del empty
            
        def t():
            _ = joblib.Parallel(n_jobs=-1)(joblib.delayed(fn)(i) for i in range(len(self.image_path_list)))
            print("resize complete")

        thread = threading.Thread(target=t)
        thread.start()

        while self.__len__() < 1:
            print("waiting resize a few images...")
            time.sleep(5)
        
    def __getitem__(self, index):
        # load image
        try:
            img_path = os.path.join(self.chache_dir, str(index) + ".jpg")
            img = Image.open(img_path)
        except Exception:
            print("Skipped error")
            img_path = os.path.join(self.chache_dir,"0.jpg")
            img = Image.open(img_path)
        # to numpy
        img = np.array(img)
        # normalize
        img = img / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
            
        return img

    def __len__(self):
        return os.listdir(self.chache_dir).__len__()

