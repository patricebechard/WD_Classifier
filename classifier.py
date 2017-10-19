#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: classifier.py
# Author: Patrice Bechard <patricebechard@gmail.com>
# Date: 01.10.2017
# Last Modified: 01.10.2017

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

actv = 2					#0: logistic, 1: tanh, 2: ReLU
optim_function = 0 			#0: AdamOptimizer, 1: Gradient descent, 2: RMS Prop, 3: AdaGrad
eta = 0.001					#learning rate

RANDOM_SEED = 123
tf.set_random_seed(RANDOM_SEED)

#--------------------------------Classes----------------------------------------




#--------------------------------Functions--------------------------------------





#--------------------------------Main-------------------------------------------

if __name__ == '__main__':
	dataset = 'preprocessed_data_copy.txt'
	data = np.loadtxt(dataset)
	print(data[0])

