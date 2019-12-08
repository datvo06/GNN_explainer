import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from chi_lib.library import *
from blocks.Block import Block


class SamplesFilter(Block):
	def __init__(self, block, sample_mask, name=None):
		Block.__init__(self, [block], name=name)
		self.block = block
		self.sample_mask = sample_mask
		self.set_mask(self.sample_mask)
		
	def implement(self):
		return {}
