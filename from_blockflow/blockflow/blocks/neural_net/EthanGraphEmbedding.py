import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from chi_lib.library import *
from model.EthanGraphCNNModel import EthanGraphCNNModel
from blocks.Block import Block


class EthanGraphEmbedding(Block):
	def __init__(self, bow_features_block, coord_features_block, adj_mat_block, name=None):
		Block.__init__(self, [bow_features_block, coord_features_block, adj_mat_block], name=name)
		self.bow_features_block = bow_features_block
		self.coord_features_block = coord_features_block
		self.adj_mat_block = adj_mat_block
		
	def implement(self):
		bow_features, _ = self.bow_features_block.get()['features']
		coord_features, _ = self.coord_features_block.get()['features']
		adj_mats, _ = self.adj_mat_block.get()['adj_mats']

		model = EthanGraphCNNModel(bow_features_tensor=bow_features, coord_features_tensor=coord_features, adj_mat_tensor=adj_mats)
		model.build()
		return {
			'features' : (model.current_V, None),
			'is_training' : (model.is_training, None),
			'global_step' : (model.global_step, None)
		}


