# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 22:59:41 2018

@author: user
"""

import chainer
import numpy as np
import chainer.functions as F

x = np.arange(1, 37).reshape(1, 1, 6, 6).astype(np.float32)
x = chainer.Variable(x)
print(x)

pooled_x, indexes = F.max_pooling_2d(x, ksize=2, stride=2, return_indices=True)
print(pooled_x)
print(indexes)

upsampled_x = F.upsampling_2d(pooled_x, indexes, ksize=2, stride=2, outsize=x.shape[2:])
print(upsampled_x.shape)
print(upsampled_x.data)

upsampled_x = F.unpooling_2d(pooled_x, ksize=2, stride=2, outsize=x.shape[2:])
print(upsampled_x.shape)
print(upsampled_x.data)

# KerasのupsamplingはChainerのunpooling
# Chainerのupsamplingはindexesがないと動かない