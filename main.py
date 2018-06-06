#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == '__main__':
	from dp import Network
	import numpy as np

	train_samples, train_labels = np.load('train_s.npy'), np.load('train_l.npy')
	test_samples, test_labels = np.load('test_s.npy'), np.load('test_l.npy')
	
	print('Training set', train_samples.shape, train_labels.shape)
	print('    Test set', test_samples.shape, test_labels.shape)



	def train_data_iterator(samples, labels, iteration_steps, chunkSize):
		'''
		Iterator/Generator: get a batch of data
		这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
		用于 for loop， just like range() function
		'''
		if len(samples) != len(labels):
			raise Exception('Length of samples and labels must equal')
		stepStart = 0  # initial step
		i = 0
		while i < iteration_steps:
			stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
			yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
			i += 1

	def test_data_iterator(samples, labels, chunkSize):
		'''
		Iterator/Generator: get a batch of data
		这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
		用于 for loop， just like range() function
		'''
		if len(samples) != len(labels):
			raise Exception('Length of samples and labels must equal')
		stepStart = 0  # initial step
		i = 0
		while stepStart < len(samples):
			stepEnd = stepStart + chunkSize
			if stepEnd < len(samples):
				yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
				i += 1
			stepStart = stepEnd


	net = Network(
		train_batch_size=128, test_batch_size=100, pooling_scale=2,
		dropout_rate = 0.90,
		base_learning_rate = 0.008, decay_rate=0.99)
	net.define_inputs(
			train_samples_shape=(128, 48, 48, 1),
			train_labels_shape=(128, 7),
			test_samples_shape=(100, 48, 48, 1),
		)
	#
	net.add_conv(patch_size=3, in_depth=1, out_depth=32, activation='relu', pooling=True, name='conv1')
	net.add_conv(patch_size=3, in_depth=32, out_depth=64, activation='relu', pooling=True, name='conv2')
	net.add_conv(patch_size=3, in_depth=64, out_depth=64, activation='relu', pooling=False, name='conv3')
	net.add_conv(patch_size=3, in_depth=64, out_depth=128, activation='relu', pooling=False, name='conv4')
	net.add_conv(patch_size=3, in_depth=128, out_depth=128, activation='relu', pooling=True, name='conv5')
	# 4 = 两次 pooling, 每一次缩小为 1/2
	# 64 = conv4 out_depth
	net.add_fc(in_num_nodes=(48 // 8) * (48 // 8) * 128, out_num_nodes=1024, activation='relu', name='fc1')
	net.add_fc(in_num_nodes=1024, out_num_nodes=256, activation='relu', name='fc2')
	net.add_fc(in_num_nodes=256, out_num_nodes=7, activation=None, name='fc3')

	net.define_model()
	#net.run(train_samples, train_labels, test_samples, test_labels, train_data_iterator=train_data_iterator, iteration_steps=3000, test_data_iterator=test_data_iterator)
	net.train(train_samples, train_labels, data_iterator=train_data_iterator, iteration_steps=3000)
	net.test(test_samples, test_labels, data_iterator=test_data_iterator)
	#net.test(val_samples, val_labels, data_iterator=test_data_iterator)

else:
    raise Exception('main.py: Should Not Be Imported!!! Must Run by "python main.py"')