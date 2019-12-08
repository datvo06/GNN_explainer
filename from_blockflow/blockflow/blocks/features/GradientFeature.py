import os, sys
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from chi_lib.library import *
from blocks.Block import Block


class GradientFeature(Block):
	def __init__(self, frozen_graph_weight_block, name=None):
		Block.__init__(self, [frozen_graph_weight_block], name=name)
		self.frozen_graph_weight_block = frozen_graph_weight_block

	def implement(self):
		self.logits, _ = self.frozen_graph_weight_block.get()['features']  # logits
		self.weights , _ = self.frozen_graph_weight_block.get()['weights']

		gradients = [tf.placeholder(dtype=tf.float32,
									shape=(1,),
									name="gradient_norm_features")
					 for w in self.weights]

		return {
			'gradients' : (gradients, None)
		}
