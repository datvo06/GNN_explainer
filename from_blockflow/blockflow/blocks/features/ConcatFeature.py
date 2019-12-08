import os, sys
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from chi_lib.library import *
from blocks.Block import Block


class ConcatFeature(Block):
	def __init__(self, feature_blocks, feature_name='features', axis=-1, name=None):
		Block.__init__(self, feature_blocksm, name=name)
		self.feature_blocks = list(feature_blocks)
		self.axis = axis
		self.feature_name = feature_name

	def implement(self):
		tensor_names = []
		tensors = []
		for block in self.feature_blocks:
			output_dict = block.get()
			pair = output_dict[self.feature_name]
			tensors.append(pair[0])

		concat_tensor = tf.concat(tensors, axis=self.axis)
		return {
			self.feature_name : (concat_tensor, None)
		}