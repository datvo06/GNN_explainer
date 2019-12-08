import os, sys
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from chi_lib.ProgressBar import ProgressBar
from chi_lib.library import *

from blocks.Block import Block


class TextlineCoordinateFeature(Block):
	def __init__(self, json_reader_block, name=None):
		Block.__init__(self, [json_reader_block], name=name)
		self.json_reader_block = json_reader_block

	def implement(self):
		_, samples = self.json_reader_block.get()['json_samples']
		samples = list(samples)		
		outputs_list = []
		logTracker.log('Bulding textline spatial features for ' + str(len(samples)) + ' samples')
		progress = ProgressBar('Bulding', len(samples))
		for sample in samples:
			outputs_list.append(get_spatial_features_matrix(sample))
			progress.increase()
		progress.done()
		features = tf.placeholder(dtype=tf.float32, shape=[None, None, outputs_list[0].shape[-1]], name='spatial_features')
		return {
			'features' : (features, outputs_list)
		}


def get_spatial_features_matrix(sample):
	def scale_non_zero(val, scale_val):
		return float(val+scale_val)/(scale_val+1.0)

	xs = []
	ys = []
	for textline in sample:
		xs.extend([p[0] for p in textline['location']])
		ys.extend([p[1] for p in textline['location']])
	
	max_x, min_x = max(xs), min(xs)
	max_y, min_y = max(ys), min(ys)	

	res = []
	for textline in sample:
		feature = np.zeros(4, dtype=np.float)
		cur_xs = [p[0] for p in textline['location']]
		cur_ys = [p[1] for p in textline['location']]
		x, y = min(cur_xs), min(cur_ys)
		w, h = max(cur_xs) - x, max(cur_ys) - y
		feature[0] = scale_non_zero((x - min_x) / (max_x - min_x), 0.1)
		feature[1] = scale_non_zero((y - min_x) / (max_y - min_y), 0.1)
		feature[2] = scale_non_zero(w / (max_x - min_x), 0.1)
		feature[3] = scale_non_zero(h / (max_y - min_y), 0.1)
		res.append(feature)
	return np.array(res, dtype=np.float32)


