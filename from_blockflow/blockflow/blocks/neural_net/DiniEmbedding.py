import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from chi_lib.library import *
from model.NonlocalModel import NonlocalModel
from blocks.Block import Block


class DiniEmbedding(Block):
	def __init__(self, bow_features_block, name=None):
		Block.__init__(self, [bow_features_block], name=name)
		self.bow_features_block = bow_features_block
		
	def implement(self):
		bow_features, _ = self.bow_features_block.get()['features']
		
		model = NonlocalModel(bow_features_tensor=bow_features)
		model.build()
		return {
			'features' : (model.current_V, None),
			'is_training' : (model.is_training, None),
			'global_step' : (model.global_step, None)
		}



