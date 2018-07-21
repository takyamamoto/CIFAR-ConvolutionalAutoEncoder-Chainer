# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 09:13:59 2018

@author: user
"""

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import numpy as np
from chainer import reporter
from PIL import Image

# Network definition
class CAE(chainer.Chain):

    def __init__(self, n_input, n_out, directory=None, return_out=False):
        super(CAE, self).__init__(
                conv1 = L.Convolution2D(None, 64, 3, pad=1),
                conv2 = L.Convolution2D(None, 128, 3, pad=1),
                conv3 = L.Convolution2D(None, 256, 3, pad=1),
                
                conv4 = L.Convolution2D(None, 256, 3, pad=1),
                conv5 = L.Convolution2D(None, 128, 3, pad=1),
                conv6 = L.Convolution2D(None, 3, 3, pad=1)
                )
        self.directory = directory
        self.return_out = return_out

    def save_image(self, arr, filename):
        img = chainer.cuda.to_cpu(arr)
        img = img * 255
        img = img.astype(np.uint8) 
        img = Image.fromarray(img)
        img.save(filename)
        
    def __call__(self, x, t):
        if self.directory is not None:
            for i in range(x.shape[0]):
                filename = self.directory + "input" + str(i) + ".png"
                #print(x.shape)
                xi = x[i]
                xi = F.transpose(xi, (1, 2, 0))
                self.save_image(xi.data, filename)
            
        # Encoder
        e = F.relu(self.conv1(x))
        e = F.max_pooling_2d(e,ksize=2, stride=2,)
        e = F.relu(self.conv2(e))
        e = F.max_pooling_2d(e,ksize=2, stride=2,)
        e_out = F.relu(self.conv3(e))
        
        # Decoder
        d = F.relu(self.conv4(e_out))
        d = F.unpooling_2d(e_out, ksize=2, stride=2, cover_all=False)
        d = F.relu(self.conv5(d))
        d = F.unpooling_2d(d, ksize=2, stride=2, cover_all=False)
        out = F.sigmoid(self.conv6(d))

        if self.directory is not None:
            for i in range(out.shape[0]):
                filename = self.directory + "output" + str(i) + ".png"
                outi = out[i]
                outi = F.transpose(outi, (1, 2, 0))
                self.save_image(outi.data, filename)
                    
        #print(out.dtype)
        #print(t.dtype)
        #print(t.shape)

        loss = F.mean_squared_error(out,t)
        
        reporter.report({'loss': loss}, self)
        
        if self.return_out == True:
            return out, loss
        else:
            return loss
        