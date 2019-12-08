import os, sys, json
import numpy as np
import tensorflow as tf

sys.path.append('../datahub/') 

from chi_lib.library import *
from blocks.Block import Block
from utils.utils import * 


class CheckpointBackup(Block):
	def __init__(self, ckp_directory_path, name=None):
		Block.__init__(self, [], name=name)
		self.ckp_directory_path = os.path.join(ckp_directory_path, 'checkpoints')
		self.saver = None

	def implement(self):
		return None

	def execute(self):
		if self.saver is None:
			# Must generate the Saver here, because Tensorflow requires building the computational graph before getting a train.Saver()
			self.saver = tf.train.Saver()

		cur_ckp_dir = os.path.join(self.ckp_directory_path, 'epoch_' + str(self.get_current_iterations()))
		createDirectory(cur_ckp_dir)
		self.saver.save(self.get_tf_session(), os.path.join(cur_ckp_dir, 'model'))
		res = {
			'log' : 'Saving checkpoint to ' + cur_ckp_dir
		}
		return res