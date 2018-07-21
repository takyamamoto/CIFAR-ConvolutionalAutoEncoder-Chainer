# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 08:51:18 2018

@author: user
"""

import argparse


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from chainer import cuda
from chainer.datasets import get_cifar10
from chainer import dataset
from chainer import Variable
from chainer import serializers
import chainer.functions as F

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
    parser.add_argument('--begin', '-b', type=int, default=0)
    args = parser.parse_args()

    # Set up a neural network to train.
    _, test_x = get_cifar10(withlabel=False, ndim=3)

    test = LoadDataset(test_x)

    model = network.CAE(3,3, return_out=True)
    
    if args.model != None:
        print( "loading model from " + args.model )
        serializers.load_npz(args.model, model)
    
    # Show 64 images
    fig = plt.figure(figsize=(6,6))
    plt.title("Original images: first rows,\n Predicted images: second rows")
    plt.axis('off')
    plt.tight_layout()
    
    pbar = tqdm(total=32)
    for i in range(4):
        for j in range(8):
            ax = fig.add_subplot(8, 8, i*16+j+1, xticks=[], yticks=[])
            x, t = test[i*8+j]
            xT = x.transpose(1, 2, 0)
            ax.imshow(xT, cmap=plt.cm.bone, interpolation='nearest')
            
            x = np.expand_dims(x, 0)
            t = np.expand_dims(t, 0)
    
            if args.gpu >= 0:
                cuda.get_device_from_id(0).use()
                model.to_gpu()
                x = cuda.cupy.array(x)
                t = cuda.cupy.array(t)
            
            predicted, loss = model(Variable(x), Variable(t))
            #print(predicted.shape)
            #print(loss)   
            
            predicted = F.transpose(predicted[0], (1, 2, 0))
            predicted = cuda.to_cpu(predicted.data) #Variable to numpy
            predicted = predicted * 255
            predicted = predicted.astype(np.uint8) 
            ax = fig.add_subplot(8, 8, i*16+j+9, xticks=[], yticks=[])
            ax.imshow(predicted, cmap=plt.cm.bone, interpolation='nearest')

            pbar.update(1)
            
    pbar.close()
   
    plt.savefig("result.png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()