import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from utils.utils import load_frozen_graph
from chi_lib.library import *
from utils.utils import * 
from chi_lib.ProgressBar import ProgressBar
from blocks.Block import Block


class ClassificationInferrer(Block):
	def __init__(self, logits_block, name=None):
		Block.__init__(self, [logits_block], name=name)
		self.logits_block = logits_block
		
	def implement(self):
		return {}

	def execute(self):
		self.logits, _ = self.logits_block.get()['features']

		information_dict = self.get()		
		feed_dict = self.get_masked_feed()

		n_samples = None
		for trainable in feed_dict:
			if n_samples is None:
				n_samples = len(feed_dict[trainable])
			if len(feed_dict[trainable]) != n_samples:
				logTracker.logException('Inconsistent sample size: ' + str(len(feed_dict[trainable])) + ' - ' + str(n_samples))

		logTracker.log('Inferring ' + str(n_samples) + ' samples')		
		session = self.get_tf_session()

		is_training = None
		if 'is_training' in information_dict:
			is_training, _ = information_dict['is_training']

		predictions = []
		progress = ProgressBar('Inferring', n_samples)
		for i in range(n_samples):
			temp_feed_dict = filter_by_indices(feed_dict, [i])
			for d in temp_feed_dict:
				temp_feed_dict[d] = np.stack(temp_feed_dict[d], axis=0)
			
			if not is_training is None:
				temp_feed_dict[is_training] = False

			output_vals = session.run([self.logits], feed_dict=temp_feed_dict)
			
			prediction = np.argmax(output_vals, axis=-1).squeeze()
			predictions.append(prediction)

			progress.increase()
		progress.done()
		return {
			'predictions' : predictions
		}


