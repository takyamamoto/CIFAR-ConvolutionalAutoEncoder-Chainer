# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 09:42:47 2018

@author: user
"""

from chainer.datasets import get_cifar10

train, test = get_cifar10(withlabel=False, ndim=3)