import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from chi_lib.ProgressBar import ProgressBar
from chi_lib.library import *

from blocks.Block import Block


class TextlineFormalKeyLabel(Block):
	def __init__(self, json_reader_block, classes, class_types, is_one_hot=False, name=None):
		Block.__init__(self, [json_reader_block], name=name)
		self.json_reader_block = json_reader_block
		classes = [c for c in classes if c != 'None']
		self.classes = list(classes)
		self.class_types = list(class_types)
		self.is_one_hot = is_one_hot
		
	def implement(self):
		_, samples = self.json_reader_block.get()['json_samples']
		samples = list(samples)

		#self.is_one_hot = True
		outputs_list = []
		logTracker.log('Building textline formal key labels for ' + str(len(samples)) + ' samples')
		progress = ProgressBar('Building', len(samples))
		for sample in samples:
			outputs_list.append(get_label_vectors(sample, self.classes, self.class_types, self.is_one_hot))
			progress.increase()
		progress.done()
		k_name = 'labels_' + str(len(self.classes)) + '_classes_' + str(mergeList(self.class_types, '_'))
		labels = tf.placeholder(dtype=tf.int32, shape=[None, None, outputs_list[0].shape[-1]], name=k_name)
		return {
			'labels'  : (labels, outputs_list),
		}
		

def get_label_vectors(sample, classes, class_types, is_one_hot):
	labels = []
	for textline in sample:
		label_info = textline['label_info']
		fk = label_info['formal_key']
		fk_type = label_info['key_type']
		if fk in classes and fk_type in class_types:
			field_idx = classes.index(fk) * len(class_types) + class_types.index(fk_type) + 1
		else:
			field_idx = 0
		labels.append([field_idx])	
	if is_one_hot:
		labels = to_one_hot(labels, (len(classes) - 1) * len(class_types) + 1)	

	return np.array(labels, dtype=np.int8)
	
