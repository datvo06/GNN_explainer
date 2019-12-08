import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from chi_lib.library import *
from utils.utils import * 
from blocks.Block import Block

class ExecutedLogExporter(Block):
	def __init__(self, blocks, res_dir, name=None):
		Block.__init__(self, blocks, name=name)
		self.blocks = list(blocks)
		self.res_dir = os.path.join(res_dir, 'executed_log')
		
	def implement(self):
		return None

	def execute(self):
		createDirectory(self.res_dir)
		logTracker.log('Exporting log of {} blocks to {}'.format(len(self.blocks), self.res_dir))
		for block in self.blocks:
			res = block.get_latest_executed()
			if not res is None:
				if 'log' in res:
					content = res['log']
				else:
					content = 'Nothing to log'
				f_name = 'epoch_' + str(self.get_current_iterations()) + '_' + block.get_name()
				with open(os.path.join(self.res_dir, f_name + '.txt'), 'w+') as f:
					f.write(content)
			else:
				logTracker.log('Nothing to export with ' + block.get_name())
		return {
			'result_directory_path' : self.res_dir
		}


