import os, sys, json, time
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from sklearn.metrics import confusion_matrix
from chi_lib.ProgressBar import ProgressBar
from chi_lib.library import *
from blocks.Block import Block
from utils.utils import * 


class AdamOptimizer(Block):
	def __init__(self, loss_block, batch_size, learning_rate, name=None):
		Block.__init__(self, [loss_block], name=name)
		self.loss_block = loss_block
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.is_initialized = False 
		self.block_dict = None
		
	def implement(self):
		temp = self.loss_block.get()
		self.losses, _ = temp['loss']
		self.accuracy, _ = temp['accuracy']
		
		global_step = None
		if 'global_step' in temp:
			global_step, _ = temp['global_step']
		self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.losses, global_step=global_step)
		return {
			'optimizer' : (self.train_op, None),
		}

	def execute(self):
		start = time.clock()
		feed_dict = self.get_masked_feed()
		session = self.get_tf_session()

		if self.block_dict is None:
			self.block_dict = self.get_value_id_to_block_dict()

		information_dict = self.get()
		
		n_samples = None
		for trainable in feed_dict:
			if n_samples is None:
				n_samples = len(feed_dict[trainable])

			if len(feed_dict[trainable]) != n_samples:
				logTracker.logException('Inconsistent sample size: ' + str(len(feed_dict[trainable])) + ' - ' + str(n_samples))

		indices = list(range(n_samples))
		np.random.shuffle(indices)
		indices_batches = get_batches(indices, batch_size=self.batch_size)

		if self.is_initialized is False:
			logTracker.log('Initialize variables')
			session.run(tf.global_variables_initializer())
			self.is_initialized = True
		
		logTracker.log('Training with ' + str(len(indices_batches)) + ' batches (Batch size: ' + str(self.batch_size) + ', total samples: ' + str(n_samples) + ')')
		exe_res = {
			'batch_losses' : [],
			'batch_accuracy' : [],
			'preprocessing_time' : time.clock() - start
		}

		start = time.clock()
		progress = ProgressBar('Training', len(indices_batches))
		for indices_batch in indices_batches:
			temp_feed_dict = filter_by_indices(feed_dict, indices_batch)
			
			#temp_feed_dict = max_padding_batch(temp_feed_dict)
			temp_feed_dict = dict(temp_feed_dict)

			# for d in temp_feed_dict:
			# 	if d is 'json':
			# 		print(temp_feed_dict[d][0][:1])

			#temp_feed_dict.pop('json')
			#temp_feed_dict.pop('paths')

			for d in temp_feed_dict:
				temp = self.block_dict[d].get_padded_batch(temp_feed_dict[d])
				temp_feed_dict[d] = np.stack(temp, axis=0)
			
			if 'is_training' in information_dict:
				is_training, _ = information_dict['is_training']
				temp_feed_dict[is_training] = True

			# features, _ = self.loss_block.get()['features']
			# labels, _ = self.loss_block.get()['labels']
			# _, res_features, res_labels, losses, accuracy = session.run([self.train_op, features, labels, self.losses, self.accuracy], feed_dict=temp_feed_dict)
			
			# prediction = np.argmax(res_features, axis=-1).squeeze().tolist()	
			# print(prediction)
			# print('-' * 30)
			# #prediction = np.argmax(res_labels, axis=-1).squeeze().tolist()	
			# res_labels = res_labels.squeeze().tolist()
			# print(res_labels)
			# print('-' * 30)
			# com = np.array(prediction) == np.array(res_labels)
			# print(com.sum())
			# print('-' * 30)
			# print(losses)
			# exit()

			_, losses, accuracy = session.run([self.train_op, self.losses, self.accuracy], feed_dict=temp_feed_dict)
			
			exe_res['batch_losses'].append(losses)
			exe_res['batch_accuracy'].append(accuracy)

			progress.increase()
		progress.done()

		exe_res['training_time'] = time.clock() - start
		exe_res['log'] = 'Avg train loss: {}, avg train accuracy: {}, training time: {}'.format(np.round(np.mean(exe_res['batch_losses']), 5), np.round(np.mean(exe_res['batch_accuracy']), 5), np.round(exe_res['training_time'], 5))
		return exe_res
