# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 08:51:18 2018

@author: user
"""

import argparse

from chainer import training
from chainer.training import extensions
from chainer import iterators, optimizers, serializers
from chainer import cuda
from chainer.datasets import get_cifar10
from chainer import dataset

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
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--noplot', dest='plot', action='store_false')
    args = parser.parse_args()

    # Set up a neural network to train.
    train_x, test_x = get_cifar10(withlabel=False, ndim=3)
    
    train = LoadDataset(train_x)
    test = LoadDataset(test_x)

    train_iter = iterators.SerialIterator(train, batch_size=args.batch, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=args.batch, repeat=False, shuffle=False)
    
    # Define model
    model = network.CAE(3,3)
    
    # Load weight
    if args.model != None:
        print( "loading model from " + args.model )
        serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(0).use()
        model.to_gpu()

    # Define optimizer
    opt = optimizers.Adam(alpha=args.lr)
    opt.setup(model)

    if args.opt != None:
        print( "loading opt from " + args.opt )
        serializers.load_npz(args.opt, opt)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, opt, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
    
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
    
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    
    # Train
    trainer.run()

    # Save results
    modelname = "./results/model"
    print( "saving model to " + modelname )
    serializers.save_npz(modelname, model)

    optname = "./results/opt"
    print( "saving opt to " + optname )
    serializers.save_npz(optname, opt) 

if __name__ == '__main__':
    main()