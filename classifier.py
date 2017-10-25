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

actv = 0					#0: logistic, 1: tanh, 2: ReLU
optim_function = 0 			#0: AdamOptimizer, 1: Gradient descent, 2: RMS Prop, 3: AdaGrad
learning_rate = 0.01
batch_size = 50
epochs = 100
n_hidden_layers = 2

RANDOM_SEED = 123
#tf.set_random_seed(RANDOM_SEED)

#--------------------------------Functions--------------------------------------

def convert_to_onehot(a,dim):

	b = np.zeros((a.shape[0],dim))

	for i in range(a.shape[-1]):
		b[i,int(a[i,0])] = 1

	return b

def forwardprop(x, w, b):
	signal = x
	for i in range(len(w)-1):
		if actv == 0:
			signal = tf.sigmoid(tf.add(tf.matmul(signal, w[i]), b[i]))
		elif actv == 1:
			signal = tf.tanh(tf.add(tf.matmul(signal, w[i]), b[i]))		
		elif actv == 2:
			signal = tf.nn.relu(tf.add(tf.matmul(signal, w[i]), b[i]))
		else:
			raise Exception ("Activation function not supported by program")	

	y_ = tf.nn.softmax(tf.add(tf.matmul(signal, w[-1]), b[-1]))
	y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)	

	return y_clipped

def backprop(cross_entropy):
	if optim_function == 0:
		optimiser = tf.train.AdamOptimizer(learning_rate = learning_rate)\
										.minimize(cross_entropy)
	elif optim_function == 1:
		optimiser = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)\
										.minimize(cross_entropy)
	elif optim_function == 2:
		optimiser = tf.train.RMSPropOptimizer(learning_rate = learning_rate)\
										.minimize(cross_entropy)
	elif optim_function == 3:
		optimiser = tf.train.AdagradOptimizer(learning_rate = learning_rate)\
										.minimize(cross_entropy)	
	else:
		raise Exception ("Optimizer not supported by program")	

	return optimiser

def train_network(train_set, valid_set,n_units, results):

	#separating inputs from labels in each dataset
	train_data_x = train_set[:,:-1]
	train_data_y = convert_to_onehot(train_set[:,-1:],n_classes)
	n_train_examples = train_data_x.shape[0]

	valid_data_x = valid_set[:,:-1]
	valid_data_y = convert_to_onehot(valid_set[:,-1:],n_classes)

	x = tf.placeholder(tf.float32, [None, n_units[0]])
	y = tf.placeholder(tf.float32, [None, n_units[-1]])

	weights = [tf.Variable(tf.random_normal([n_units[i],n_units[i+1]],stddev=0.03))\
									for i in range(len(n_units)-1)]

	bias = [tf.Variable(tf.random_normal([n_units[i]]))\
									for i in range(1,len(n_units))]

	"""
	W1 = tf.Variable(tf.random_normal([n_inputs, n_hidden_units], stddev=0.03), name='W1')
	b1 = tf.Variable(tf.random_normal([n_hidden_units]), name='b1')

	W2 = tf.Variable(tf.random_normal([n_hidden_units, n_classes], stddev=0.03), name='W2')
	b2 = tf.Variable(tf.random_normal([n_classes], name='b2'))
	"""

	y_clipped = forwardprop(x, weights, bias)

	cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)\
									+ (1 - y) * tf.log(1 - y_clipped), axis=1))

	optimiser = backprop(cross_entropy)

	init_op = tf.global_variables_initializer()

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_clipped, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		#initialize the variables
		sess.run(init_op)
		total_batch = int( n_train_examples / batch_size)
		for epoch in range(epochs):
			avg_cost = 0
			for i in range(total_batch):
				batch_x = train_data_x[i*batch_size:(i+1)*batch_size,:]
				batch_y = train_data_y[i*batch_size:(i+1)*batch_size,:]

				_, c = sess.run([optimiser, cross_entropy],
								feed_dict={x: batch_x, y: batch_y})
				avg_cost += c / total_batch
			if epoch % 10 == 0:
				print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
		acc_results = sess.run(accuracy, feed_dict={x: valid_data_x, y: valid_data_y})
		print(acc_results)
		results.write(str(acc_results)+'\n\n')

#--------------------------------Main-------------------------------------------

if __name__ == '__main__':

	results = open('log.txt','w')
	results.write('number of hidden layers : %d\n'%n_hidden_layers)

	#dataset = 'prepared_data.save'
	dataset = 'test.csv'
	data = np.loadtxt(dataset,delimiter=',')
	data.astype('float32')
	np.random.shuffle(data)

	n_examples = data.shape[0]
	n_inputs = data.shape[-1]-1
	# labels : ['DA','DB','DO','DC','DQ','DZ','DAB','DAZ','DA+M','DAM','DAH','DBZ']
	n_classes = 12

	#separating dataset in training, validation and test sets
	train_set = data[:n_examples//2, :]
	valid_set = data[(n_examples)//2 : (3*n_examples)//4, :]
	test_set = data[(3*n_examples)//4:]

	if n_hidden_layers == 1:
		for n_hidden_units in np.arange(20,301,20):

			n_units = []
			n_units.append(n_inputs)
			n_units.append(n_hidden_units)
			n_units.append(n_classes)
			results.write('number of hidden units : %d\n'%n_hidden_units)

			print("Number of hidden units : %d"%n_hidden_units)
			train_network(train_set, valid_set, n_units, results)

	elif n_hidden_layers == 2:
		for n_hidden_units_1 in np.arange(20,301,20):
			for n_hidden_units_2 in np.arange(20,n_hidden_units_1+1,10):
				n_units = []
				n_units.append(n_inputs)
				n_units.append(n_hidden_units_1)
				n_units.append(n_hidden_units_2)
				n_units.append(n_classes)
				results.write('Number of hidden units : %d %d\n'%\
										(n_hidden_units_1, n_hidden_units_2))

				print("Number of hidden units : %d %d"%\
											(n_hidden_units_1, n_hidden_units_2))
				train_network(train_set, valid_set, n_units, results)
	else:
		raise Exception ("only one or two layers are supported")

	results.close()

