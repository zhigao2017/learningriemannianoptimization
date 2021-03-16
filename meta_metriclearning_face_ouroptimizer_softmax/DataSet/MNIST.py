from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
#from PIL import Image

import os
import numpy as np
import gzip
import pickle

from collections import defaultdict

#from sklearn.decomposition import PCA


class MNIST(data.Dataset):
    def __init__(self, filepath=None,train=True):

        # Initialization data path and train(gallery or query) txt path

        self.filepath=filepath
        #self.train_data,self.train_labels,self.test_data,self.test_labels=self.load(self.filepath)

        self.train_data=np.load(filepath+'/dim784_training_images.npy')
        self.train_labels=np.load(filepath+'/dim784_training_labels.npy')
        self.test_data=np.load(filepath+'/dim784_test_images.npy')
        self.test_labels=np.load(filepath+'/dim784_test_labels.npy')

        #X=np.append(self.train_data,self.test_data,axis=0)
        #pca = PCA(n_components=128, svd_solver='full')
        #X=pca.fit_transform(X)
        #self.train_data=X[0:self.train_labels.shape[0],:]
        #self.test_data=X[self.train_labels.shape[0]:self.train_labels.shape[0]+self.test_labels.shape[0],:]

        #print('X shape',X.shape)

        if train==True:
            self.data=self.train_data
            self.labels=self.train_labels.tolist()
        else:
            self.data=self.test_data
            self.labels=self.test_labels.tolist()

        self.data=self.data.astype(np.float32)
        #self.data=self.data/255

        #self.data=self.data.tolist()
        #self.labels=self.labels.tolist()

        Index = defaultdict(list)
        for i, label in enumerate(self.labels):
            Index[label].append(i)

        self.Index = Index

        classes = list(set(self.labels))
        self.classes = classes



    def load(self,filepath):
        print('filepath',filepath)
        with open(filepath,'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]        

    def __getitem__(self, index):

        img, label = self.data[index], self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)






