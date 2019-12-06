import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from normalizing.normalize_text import normalize_text
from chi_lib.ProgressBar import ProgressBar
from chi_lib.library import *

from blocks.Block import Block


class FormBowFeature(Block):
	def __init__(self, json_reader_block, corpus, name=None):
		Block.__init__(self, [json_reader_block], name=name)
		self.json_reader_block = json_reader_block
		self.corpus = corpus
		
	def implement(self):
		_, samples = self.json_reader_block.get()['json_samples']
		samples = list(samples)

		char_to_idx = {c : i for i, c in enumerate(self.corpus)}
		outputs_list = []
		logTracker.log('Building BoW features for ' + str(len(samples)) + ' samples')
		progress = ProgressBar('Building', len(samples))
		for sample in samples:
			outputs_list.append(get_bow_matrix(char_to_idx, sample))
			progress.increase()
		progress.done()
		features = tf.placeholder(dtype=tf.float32, shape=[None, None, outputs_list[0].shape[-1]], name='bow_features')
	
		return {
			'features' : (features, outputs_list)
		}
		

def get_bow_matrix(char_to_idx, sample):
	res = []
	for textline in sample:
		res.append(get_text_bow(textline['text'], char_to_idx))
	return np.array(res, dtype=np.float32)


def get_text_bow(text, char_to_idx):
	res = np.zeros(len(char_to_idx), dtype=np.float)
	text = normalize_text(text)
	for c in text:
		if c not in char_to_idx:
			continue
		res[char_to_idx[c]] = 1
	return res
