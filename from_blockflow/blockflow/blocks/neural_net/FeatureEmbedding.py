import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from model.graph.graphcnn import layers
from chi_lib.library import *
from blocks.Block import Block


class FeatureEmbedding(Block):
	def __init__(self, features_block, output_feature_size, axis=-1, name=None):
		Block.__init__(self, [features_block], name=name)
		self.features_block = features_block
		self.output_feature_size = output_feature_size
		self.axis = axis
		
	def implement(self):
		features, _ = self.features_block.get()['features']
		embedded_features = layers.make_embedding_layer_on_axis(features, self.output_feature_size, axis=self.axis)
		return {
			'features' : (embedded_features, None)
		}


