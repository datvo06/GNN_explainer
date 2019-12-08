import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from chi_lib.library import *
from utils.utils import * 
from blocks.Block import Block

class BlocksRunner(Block):
	def __init__(self, runnable_blocks, runnable_delays=None, iterations=1, print_log=True, name=None):
		Block.__init__(self, runnable_blocks, name=name)
		self.runnable_blocks = list(runnable_blocks)
		if runnable_delays is None:
			self.runnable_delays = [1] * len(runnable_blocks)
		else:
			self.runnable_delays = list(runnable_delays)
		self.iterations = iterations
		self.print_log = print_log
		
	def implement(self):
		return None

	def execute(self):
		for _ in range(1, self.iterations + 1):
			Block.increase_current_iteration()
			current_iter = Block.get_current_iterations()
			logTracker.log('Iteration {} / {}'.format(Block.get_current_iterations(), self.iterations))
			logTracker.log('Running {} blocks'.format(len(self.runnable_blocks)))
			for block_idx, block in enumerate(self.runnable_blocks):
				if current_iter % self.runnable_delays[block_idx] != 0:
					continue

				temp = block.run()
				if not temp is None and self.print_log is True:
					temp = dict(temp)
					mess = 'No returned log'
					if 'log' in temp:
						mess = temp.pop('log')
					logTracker.log(mess + '\n')
			logTracker.log('-' * 30)
		return None


