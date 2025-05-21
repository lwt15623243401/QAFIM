# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform



import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np

class INatDataset(ImageFolder):
    #加载数据
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        '''
        root：数据集的根目录，也就是存放 iNaturalist 数据集文件的主文件夹路径。
        train：布尔类型，默认值为 True，用于指定加载的是训练集（True）还是验证集（False）。
        year：整数类型，默认值为 2018，代表数据集的年份版本。
        transform：图像预处理操作，默认为 None。可以传入 PyTorch 中的 torchvision.transforms 模块里的一系列变换操作，用于对图像进行如缩放、裁剪、归一化等处理。
        target_transform：标签预处理操作，默认为 None，用于对标签数据进行特定的变换。
        category：字符串类型，默认值为 'name'，表示使用的分类类别层级，可选值包括 'kingdom'、'phylum'、'class'、'order'、'supercategory'、'family'、'genus'、'name' 等。
        loader：图像加载器，默认为 default_loader，用于从文件系统中读取图像数据。
        '''
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

#原代码
#这段代码定义了一个名为 build_dataset 的函数，其主要功能是根据传入的参数构建不同的数据集对象，并返回数据集对象以及该数据集所包含的类别数量。这个函数可以根据用户指定的数据集类型（如 CIFAR、IMNET、INAT 等）和训练 / 验证状态，动态地创建相应的数据集实例
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


#build_transform 函数根据训练状态和输入图像大小，构建了不同的图像预处理变换操作，以满足训练和验证 / 测试阶段的需求。在训练阶段，通常会使用更多的数据增强操作来提高模型的泛化能力；在验证或测试阶段，主要进行图像的调整大小、裁剪和归一化操作。
def build_transform(is_train, args):
    resize_im = args.input_size > 32
    '''
    作用：判断输入图像尺寸是否需要调整（通常 32 是 CIFAR 等小尺寸数据集的默认尺寸，大于 32 时需 Resize）。
    逻辑：若 input_size > 32（如 224），resize_im 为 True，后续会使用 Resize 操作；否则（如 32）为 False，使用 RandomCrop。
    '''
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = args.input_size if args.input_size > 224 else int((256 / 224) * args.input_size)
        # size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)





