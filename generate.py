# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 08:51:18 2018

@author: user
"""

import argparse

from chainer import serializers

import numpy as np
from chainer import cuda
from chainer.datasets import get_cifar10
from chainer import dataset
from chainer import Variable
import network

# Load data
class LoadDataset(dataset.DatasetMixin):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        return self.data[i], self.data[i]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, default="./results/model")
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--batch', '-b', type=int, default=32)
    args = parser.parse_args()

    # Set up a neural network to train.
    _, test_x = get_cifar10(withlabel=False, ndim=3)

    test = LoadDataset(test_x)

    model = network.CAE(3,3, directory="img/", return_out=True)
    
    if args.model != None:
        print( "loading model from " + args.model )
        serializers.load_npz(args.model, model)
    
    idx = 0
    x, t = test[idx]
    
    x = np.expand_dims(x, 0)
    t = np.expand_dims(t, 0)

    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        model.to_gpu()
        x = cuda.cupy.array(x)
        t = cuda.cupy.array(t)
    
    predicted, loss = model(Variable(x), Variable(t)) 
    print(predicted.shape)
    print(loss)   

if __name__ == '__main__':
    main()