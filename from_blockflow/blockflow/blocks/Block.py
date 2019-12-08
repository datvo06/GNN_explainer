import os, sys, json
import time
import shutil
import numpy as np
import regex as re
import tensorflow as tf
sys.path.append('../datahub/') 

from normalizing.normalize_text import normalize_text
from copy import deepcopy
from collections import defaultdict
from chi_lib.ProgressBar import ProgressBar
from chi_lib.library import *
from utils.utils import *


class Block:
	__iterations = 0
	__session = tf.Session()
	cached_class_names = set()
	cached_output_dicts = {}
	def __init__(self, input_blocks, name):
		self.name = None
		self.name = self.generate_class_name(name)
		self.input_blocks = list(input_blocks)
		self.current_information_dict = None
		self.wait_for_resolving_conflict = None
		self.output_dict = None
		self.__latest_executed = None
		self.mask = None
		
	def implement(self):
		raise NotImplementedError('Not implemented "implement"')

	def execute(self):	
		# If it is overriden in derived classes, it means those classes are executable blocks
		raise NotImplementedError('Not implemented "execute"')

	def get_feed(self):
		# The feeding tensors can be different from the combined information tensors
		feed_dicts = []
		for input_block in self.input_blocks:
			feed_dicts.append(input_block.get_feed())

		res = {}
		for feed_dict in feed_dicts:
			for trainable, data in feed_dict.items():
				# We only keep the one which exists both the tensor (variable, placeholder) and the feeding value
				if not trainable is None and not data is None:
					if trainable in res:
						if res[trainable] != data:
							logTracker.logException('Conflicting trainable feeding tensor: ' + str(trainable))
					res[trainable] = data

		# If the data flow is modified, then keep the latest one
		# Simply, just replace all the data for this block

		for _, (trainable, data) in self.output_dict.items():
			if not trainable is None and not data is None:
				res[trainable] = data
		return res

	def get_masked_feed(self):
		feed_dict = self.get_feed()
		# Filtering the samples
		mask = self.get_mask()
		for trainable in feed_dict:
			feed_dict[trainable] = filter_by_mask(feed_dict[trainable], mask)
		return feed_dict

	# This method can be overriden to customize the padding process
	def get_padded_batch(self, batch_values):
		max_dim_dict = defaultdict(lambda : 0)
		cur_rank = None
		for value in batch_values:
			cur_shape = value.shape
			if cur_rank is None:
				cur_rank = len(cur_shape)
			if len(cur_shape) != cur_rank:
				logTracker.logException('Inconsistent feeding value ranks: ' + str(len(cur_shape)) + ' - ' + str(cur_rank))
			for i, dim in enumerate(cur_shape):
				max_dim_dict[i] = max(max_dim_dict[i], dim)

		res = []
		for value in batch_values:
			value = value.astype(np.float32)
			cur_shape = value.shape
			padding_info = []
			for r in range(cur_rank):	
				padding_info.append((0, max_dim_dict[r] - cur_shape[r]))
			padding_info = tuple(padding_info)
			res.append(np.pad(value, padding_info, 'constant', constant_values=(0)))
		return res

	@staticmethod
	def increase_current_iteration():
		Block.__iterations += 1

	@staticmethod
	def get_current_iterations():
		return Block.__iterations

	@Access.final
	def get_name(self):
		return self.name	

	@Access.final
	def get_tf_session(self):
		return Block.__session

	@Access.final	
	def set_tf_session(self, session):
		Block.__session = session
		Block.__session.as_default()

	@Access.final
	def generate_default_mask(self):
		if not self.mask is None:
			return
		self.generate_input_blocks_information_dict()
		for info_name in self.current_information_dict:
			trainable, data = self.current_information_dict[info_name]
			if not data is None:
				if self.mask is None:
					# By default, all samples are selected
					self.mask = [True] * len(data)
				else:
					if len(data) != len(self.mask):
						logTracker.logException('Inconsistent feeding data size and mask size: {} - {}'.format(len(data), len(self.mask)))

		masks = []
		for input_block in self.input_blocks:
			cur_mask = input_block.get_mask()
			if not cur_mask is None:
				masks.append(cur_mask)

		if not self.mask is None:
			masks.append(self.mask)
		self.mask = and_masks(masks)

	@Access.final
	def get_mask(self):
		self.generate_default_mask()
		if self.mask is None:
			logTracker.log('Not found mask for ' + self.get_name())
			return None 
		return list(self.mask)

	@Access.final
	def set_mask(self, new_mask):
		self.generate_default_mask()
		if len(self.mask) != len(new_mask):
			logTracker.logException('Inconsistent old mask size and new mask size: {} - {}'.format(len(new_mask), len(self.mask)))
		logTracker.log('Set new mask for ' + self.get_name())
		self.mask = list(new_mask)

	@Access.final
	def get_value_id_to_block_dict(self):
		block_dicts = []
		for input_block in self.input_blocks:
			block_dicts.append(input_block.get_value_id_to_block_dict())

		res = {}
		for block_dict in block_dicts:
			for trainable, block in block_dict.items():
				if not trainable is None:
					if trainable in res:
						if res[trainable] != block:
							logTracker.logException('Conflicting trainable to block dict: ' + str(trainable))
					res[trainable] = block

		for _, (trainable, data) in self.output_dict.items():
			if not trainable is None:
				if trainable in res:
					if res[trainable] != block:
						logTracker.logException('Conflicting trainable to block dict: ' + str(trainable))
				res[trainable] = self
		return res	

	@Access.final
	def get_latest_executed(self):
		return deepcopy(self.__latest_executed)

	@Access.final
	def run(self):
		# Let's build first
		self.get()
		# then execute
		self.__latest_executed = self.execute()
		return self.__latest_executed

	@Access.final
	def build(self):
		# Do not call build directly, please use 'get' instead
		# We need to set the current session's graph to default, so that whenever you call 
		with self.get_tf_session().graph.as_default():
			with tf.variable_scope(self.name) as scope:
				self.output_dict = self.implement()
				self.generate_default_mask()
		return self.output_dict

	@Access.final
	def get(self):
		self.generate_input_blocks_information_dict()
		if not self in Block.cached_output_dicts:
			output_dict = self.build()
			if not output_dict is None:
				# Check the resolving results first
				for info_name in self.wait_for_resolving_conflict:
					if info_name in output_dict:
						logTracker.log('Resolved trainable conflict from "' + str(info_name) + '":')
						logTracker.log('\t- Input conflict : ' + str(list(self.wait_for_resolving_conflict[info_name])))
						logTracker.log('\t- Joint to       : ' + str(output_dict[info_name]))
					else:
						logTracker.logException('[' + self.get_name() + '] No resolve for trainable conflicts from "' + str(info_name) + '": ' + '\t- ' + str(self.wait_for_resolving_conflict[info_name]))
				# Updated the aggregated information by info name, after building this block
				# It means we combine the information of input blocks and the newly created block
				for info_name in output_dict:
					self.current_information_dict[info_name] = output_dict[info_name]	
				Block.cached_output_dicts[self] = self.current_information_dict
			else:
				logTracker.log('Reached terminating block: ' + str(self))

		elif Block.cached_output_dicts[self] != self.current_information_dict:
			logTracker.logException('Inconsistent cached current_output_dict: \n' + str(Block.cached_output_dicts[self]) + '\n\n' + str(self.current_information_dict))

		return self.current_information_dict

	@Access.final
	def generate_class_name(self, current_name):
		if not self.name is None:
			return self.name
			
		if current_name is None:
			current_name = self.__class__.__name__
			temp_name = current_name
			i = 0
			while temp_name in Block.cached_class_names:
				temp_name = str(current_name) + '_' + str(i)
				i += 1
			current_name = temp_name
		if current_name in Block.cached_class_names:
			logTracker.logException('Existing class name for variable scope: ' + str(current_name))
		Block.cached_class_names.add(current_name)
		return current_name

	@Access.final
	def generate_input_blocks_information_dict(self):
		if self.current_information_dict is None or self.wait_for_resolving_conflict is None:
			# Calculate the information flow from the input blocks
			self.current_information_dict, self.wait_for_resolving_conflict = compute_information_dict(self.input_blocks)


def compute_information_dict(blocks):
	current_information_dict = {}
	wait_for_resolving_conflict = defaultdict(lambda : set())
	for input_block in blocks:
		current_output_dict = input_block.get()
		for info_name, pair in current_output_dict.items():
			if info_name in current_information_dict:
				# Checking the conflict
				existing_pair = current_information_dict[info_name]
				if len(existing_pair) != len(pair):
					logTracker.logException('Conflicting existing information name, with different pair length: ' + str(info_name) + ' - ' + str(len(existing_pair)) + ' - ' + str(len(pair)) + ' in block ' + str(input_block.get_name()) + ' with info "' + info_name + '"')
				
				cur_trainable, cur_data = pair
				existing_trainable, existing_data = existing_pair

				# If duplicated information name, but from 2 different flows (2 different tensor flows), then we wait for resolving in this block.
				if cur_trainable != existing_trainable:
					wait_for_resolving_conflict[info_name].add(cur_trainable)
					wait_for_resolving_conflict[info_name].add(existing_trainable)
					logTracker.log('Wait for resolving trainable conflict from "' + str(info_name) + '": ')
					logTracker.log('\t- ' + str(cur_trainable))
					logTracker.log('\t- ' + str(existing_trainable))
				else:
					# If duplicated information name and from the sample flow and on the same trainable, but with different feeding data, then we raise an Error due to inconsistent.
					if cur_data != existing_data:
						logTracker.logException('Conflicting existing information name, with different feeding data for the same trainable joint in this block: ' + str(cur_trainable) + ' - ' + str(existing_trainable) + ' in block ' + str(input_block.get_name()) + ' with info "' + info_name + '"')
			else:
				# Otherwise, keep it as a single information flow	
				current_information_dict[info_name] = pair
	return current_information_dict, wait_for_resolving_conflict

		