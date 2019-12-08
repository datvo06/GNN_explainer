import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from chi_lib.ProgressBar import ProgressBar
from chi_lib.library import *

from blocks.Block import Block
from model.graph.graph_utils import Graph


class HeuristicGraphAdjMat(Block):
	def __init__(self, json_reader_block, name=None):
		Block.__init__(self, [json_reader_block], name=name)
		self.json_reader_block = json_reader_block
		self.num_edge = 6
		
	def implement(self):
		_, samples = self.json_reader_block.get()['json_samples']
		samples = list(samples)
		
		outputs_list = []
		logTracker.log('Building heuristic graph for ' + str(len(samples)) + ' samples')
		progress = ProgressBar('Building', len(samples))
		for sample in samples:
			outputs_list.append(get_heuristic_graph_adj_mat(sample))
			progress.increase()
		progress.done()
		adj_mats = tf.placeholder(dtype=tf.float32, shape=[None, None, self.num_edge, None], name='heuristic_adj_mat')

		return {
			'adj_mats' : (adj_mats, outputs_list)
		}
		

def get_heuristic_graph_adj_mat(sample):
	sample = convert_to_old_format(sample)
	g = Graph(sample, None)
	return g.adj


def convert_to_old_format(standard_format_sample):
	def get_x_y_w_h(loc):
		xs = [p[0] for p in loc]
		ys = [p[1] for p in loc]
		min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
		return min_x, min_y, max_x - min_x, max_y - min_y

	res = {}
	for i, tl in enumerate(standard_format_sample):
		label = 'None'
		if 'label_info' in tl:
			label = tl['label_info']['formal_key']

		res['text_line' + str(i)] = {
			"value": tl['text'], 
			"location": get_x_y_w_h(tl['location']), 
			"label": label
		}
	return res
