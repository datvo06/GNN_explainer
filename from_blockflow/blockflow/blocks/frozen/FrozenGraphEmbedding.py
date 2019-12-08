import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from utils.utils import load_frozen_graph
from chi_lib.library import *
from utils.utils import * 
from chi_lib.ProgressBar import ProgressBar
from blocks.Block import Block


class FrozenGraphEmbedding(Block):
	def __init__(self, bow_features_block, coord_features_block, adj_mat_block, model_pb_path, name=None):
		Block.__init__(self, [bow_features_block, coord_features_block, adj_mat_block], name=name)
		self.bow_features_block = bow_features_block
		self.coord_features_block = coord_features_block
		self.adj_mat_block = adj_mat_block
		self.model_pb_path = model_pb_path
		
	def implement(self):
		self.bow_features, _ = self.bow_features_block.get()['features']
		self.coord_features, _ = self.coord_features_block.get()['features']
		self.adj_mats, _ = self.adj_mat_block.get()['adj_mats']

		graph = load_frozen_graph(self.model_pb_path)
		self.set_tf_session(tf.Session(graph=graph))
		#graph_def = graph.as_graph_def()
		
		self.input_bow_tensor = graph.get_tensor_by_name('import/{}:0'.format('inferrence_bow_features'))
		self.input_coord_tensor = graph.get_tensor_by_name('import/{}:0'.format('inferrence_coord_features'))
		self.input_adj_tensor = graph.get_tensor_by_name('import/{}:0'.format('inferrence_adj_mats'))
		self.output_tensor = graph.get_tensor_by_name('import/{}:0'.format('output_features'))
		self.is_training = graph.get_tensor_by_name('import/{}:0'.format('is_training'))
		self.graph = graph
		return {
			'features' : (self.output_tensor, None),
			'is_training' : (self.is_training, None),	
		}

	def get_feed(self):
		feed_dict = super().get_feed()
		replace_dict = {
			self.bow_features : self.input_bow_tensor,
			self.coord_features : self.input_coord_tensor,
			self.adj_mats : self.input_adj_tensor,
		}

		for cur_tensor in replace_dict:
			if cur_tensor in feed_dict:
				new_tensor = replace_dict[cur_tensor]
				if new_tensor in feed_dict:
					logTracker.logException('Duplicated tensor node: ' + str(new_tensor))
				feed_dict[new_tensor] = feed_dict.pop(cur_tensor)
		return feed_dict

	def execute(self):
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

		predictions = []
		progress = ProgressBar('Inferring', n_samples)
		for i in range(n_samples):
			temp_feed_dict = filter_by_indices(feed_dict, [i])
			for d in temp_feed_dict:
				temp_feed_dict[d] = np.stack(temp_feed_dict[d], axis=0)
			temp_feed_dict[self.is_training] = False

			output_vals = session.run([self.output_tensor], feed_dict=temp_feed_dict)
			
			prediction = np.argmax(output_vals, axis=-1).squeeze()
			predictions.append(prediction)

			progress.increase()
		progress.done()
		return {
			'predictions' : predictions
		}


