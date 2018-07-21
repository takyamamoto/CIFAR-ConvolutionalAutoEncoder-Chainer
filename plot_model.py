# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 08:51:18 2018

@author: user
"""

import chainer.computational_graph as c
import numpy as np
from chainer import Variable
import network

def plot_model():
    model = network.CAE(3,3, return_out=True)
    
    # Draw Network Graph
    features = np.empty((1, 3, 32,32), dtype=np.float32)
    x = Variable(features)
    y = model(x, x)
    
    g = c.build_computational_graph(y)
    with open('graph.dot', 'w') as o:
        o.write(g.dump())

if __name__ == '__main__':
    plot_model()