import os, sys, json, re
import numpy as np
import tensorflow as tf
 

from datahub.normalizing.normalize_text import normalize_text
from blockflow.utils.utils import load_frozen_graph
from datahub.chi_lib.library import *
from blockflow.utils.utils import * 
from datahub.chi_lib.ProgressBar import ProgressBar
from blockflow.blocks.Block import Block


class OneToOneTextlineKeyValueInferrer(Block):
	def __init__(self, predictions_inferrer_block, fileloader_block, classes, corpus, name=None):
		Block.__init__(self, [predictions_inferrer_block, fileloader_block], name=name)
		self.predictions_inferrer_block = predictions_inferrer_block
		self.fileloader_block = fileloader_block
		self.classes = convert_to_key_value_classes(classes)
		self.corpus = corpus
		
	def implement(self):
		return {}

	def execute(self):
		predictions = self.predictions_inferrer_block.get_latest_executed()['predictions']
		# softmax_scores = self.predictions_inferrer_block.get_latest_executed()['softmax_scores']
		_, json_samples = self.fileloader_block.get()['json_samples']
		_, json_file_paths = self.fileloader_block.get()['json_file_paths']

		mask = self.get_mask()
		json_samples = filter_by_mask(json_samples, mask)
		json_file_paths = filter_by_mask(json_file_paths, mask)

		if len(json_samples) != len(predictions):
			logTracker.logException('Inconsistent number of samples and predictions: {} - {}'.format(len(json_samples), len(predictions)))

		logTracker.log('Inferring textline for ' + str(len(json_samples)) + ' samples')		
		
		outputs = []
		corpus_cleaner = re.compile(r'[^' + re.escape(self.corpus) + r']')
		progress = ProgressBar('Inferring', len(json_samples))
		for i, sample in enumerate(json_samples):
			cur_prediction = predictions[i].tolist()
			# cur_softmax = softmax_scores[i].tolist()
			cur_input = deepcopy(sample)
			if len(cur_prediction) != len(sample):
				logTracker.logException('Inconsistent number of textlines and textline predictions: {} - {}'.format(len(cur_prediction), len(sample)))

			for j, textline in enumerate(sample):
				cur_type = self.classes[cur_prediction[j]]
				try:
					cur_key_type, cur_type = cur_type.split("#")
				except ValueError:
					cur_key_type = 'other'
				cur_input[j]['key_type'] = cur_key_type
				cur_input[j]['type'] = cur_type
				# cur_input[j]['confidence'] = round(float(cur_softmax[j]), 4)
				norm_text = normalize_text(cur_input[j]['text'])
				norm_text = corpus_cleaner.sub('�', norm_text)
				cur_input[j]['infer_info'] = {
					'norm_text' : norm_text,
					'bow' : ''.join(sorted(set(norm_text)))
				}
			outputs.append(cur_input)
			progress.increase()
		progress.done()
		outputs = list(zip(json_file_paths, outputs))
		return {
			'json_outputs' : outputs
		}


def convert_to_key_value_classes(classes):
	classes = [c for c in classes if c != 'None']
	res = ['None']
	for c in classes:
		res.append('key#' + c)
		res.append('value#' + c)
	return res