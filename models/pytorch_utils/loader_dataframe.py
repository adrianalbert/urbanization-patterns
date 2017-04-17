# Adrian Albert
# 2017
# adapted from 
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py

import torch.utils.data as data
import torch

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
    if df[classCol].dtype in [float, np.float32, np.float64]:
        classes = None
        class_to_idx = None
    else:
        classes = df[classCol].unique().tolist()
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(df, class_to_idx, classCol="class", filenameCol="filename"):
    images = []
    df = df[df[filenameCol].apply(is_image_file)]
    if class_to_idx is not None:
        labels = df[classCol].apply(lambda x: class_to_idx[x]).values.tolist()
    else:
        labels = df[classCol].values.tolist()
    images = zip(df[filenameCol].values.tolist(), 
                 labels)
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
        classCol = kwargs['classCol'] if 'classCol' in kwargs else 'class'
        classes, class_to_idx = find_classes(df, classCol=classCol)
        imgs = make_dataset(df, class_to_idx, classCol=classCol)
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


class WeightedRandomSampler(data.sampler.Sampler):
    """Samples elements from [0,..,len(weights)-1] with given probabilities (weights).
    Arguments:
        weights (list)   : a list of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
    """

    def __init__(self, weights, num_samples, replacement=True):
        self.weights = torch.DoubleTensor(weights)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples

class BalancedRandomSampler(data.sampler.Sampler):
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, sources, labels, bal=1):
        self.num_samples = len(sources)
        self.bal = bal
        self.labels = labels

    def __iter__(self):
        self.labels
        return iter(torch.randperm(self.num_samples).long())

    def __len__(self):
        return self.num_samples
