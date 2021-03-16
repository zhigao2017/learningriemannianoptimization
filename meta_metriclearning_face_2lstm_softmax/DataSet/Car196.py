from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
from PIL import Image

import os
from torchvision import transforms
from collections import defaultdict


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyData(data.Dataset):
    def __init__(self, root=None, label_txt=None,
                 transform=None, loader=default_loader):

        # Initialization data path and train(gallery or query) txt path

        if label_txt is None:
            root = "/data/xunwang/DataSet/Car196/"
            label_txt = os.path.join(root, 'train.txt')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if transform is None:
            transform = transforms.Compose([
                # transforms.CovertBGR(),
                transforms.Resize(256),
                transforms.RandomResizedCrop(scale=(0.16, 1), size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        # read txt get image path and labels
        file = open(label_txt)
        images_anon = file.readlines()

        images = []
        labels = []

        for img_anon in images_anon:
            # img_anon = img_anon.replace(' ', '\t')

            [img, label] = img_anon.split(' ')
            images.append(img)
            labels.append(int(label))

        classes = list(set(labels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels):
            Index[label].append(i)

        # Initialization Done
        self.root = root
        self.images = images
        self.labels = labels
        self.classes = classes
        self.transform = transform
        self.Index = Index
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.images[index], self.labels[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)


class Car196:
    def __init__(self, root=None, transform=None):
        # Data loading code

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if transform is None:
            transform = [transforms.Compose([
                # transforms.CovertBGR(),
                transforms.Resize(256),
                transforms.RandomResizedCrop(scale=(0.16, 1), size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
                transforms.Compose([
                    # transforms.CovertBGR(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])]

        root_ = '/data/xunwang/DataSet/Car196/'

        train_txt = os.path.join(root_, 'train.txt')
        test_txt = os.path.join(root_, 'test.txt')

        self.train = MyData(root, label_txt=train_txt, transform=transform[0])
        self.test = MyData(root, label_txt=test_txt, transform=transform[1])


def testCar196():
    data = Car196()
    print(len(data.test))
    print(len(data.train))


if __name__ == "__main__":
    testCar196()


