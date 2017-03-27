# Adrian Albert
# 2017
# adapted from 
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py

import torch.utils.data as data

from PIL import Image
import os
import os.path
import pandas as pd
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(df, classCol="class"):
    classes = df[classCol].unique().tolist()
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(df, class_to_idx, classCol="class", filenameCol="filename"):
    images = []
    df = df[df[filenameCol].apply(is_image_file)]
    images = zip(df[filenameCol].values.tolist(), 
                 df[classCol].apply(lambda x: class_to_idx[x]).values.tolist())
    return images


def default_loader(path, mode="RGB"):
    '''
        mode can be either "RGB" or "L" (grayscale)
    '''
    return Image.open(path).convert(mode)


def grayscale_loader(path, val_nodata=128):
    pimg = default_loader(path, mode="L")
    img = np.array(pimg)
    img[abs(img-val_nodata)<0.01] = 0 # hack to remove no-data patches
    pimg = Image.fromarray(np.uint8(img))    
    return pimg


class ImageDataFrame(data.Dataset):
    '''
    Assumes a Pandas dataframe input with the following columns:
        filename, class
    '''

    def __init__(self, df, transform=None, target_transform=None,
                 loader=default_loader, **kwargs):
        classes, class_to_idx = find_classes(df)
        imgs = make_dataset(df, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in dataframe of: "+len(df)+"\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.df = df
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)