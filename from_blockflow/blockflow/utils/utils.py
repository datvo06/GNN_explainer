import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 
from chi_lib.library import *


def load_corpus(corpus_path):
	return load_json(corpus_path)


def load_classes(classes_path):
	classes = load_json(classes_path)
	if not 'None' in classes:
		classes = ['None'] + classes
	return classes


def load_frozen_graph(pb_path):
	with tf.gfile.GFile(pb_path, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(
			graph_def, 
			input_map=None, 
			return_elements=None, 
			op_dict=None, 
			producer_op_list=None
		)
		
	return graph


def and_masks(masks):
	res_mask = None
	for mask in masks:
		if res_mask is None:
			res_mask = list(mask)
		elif len(mask) != len(res_mask):
			logTracker.logException('Inconsistent mask sizes: {} - {}'.format(len(res_mask), len(mask)))
		res_mask = [True if mask[i] and res_mask[i] else False for i in range(len(res_mask))]
	return res_mask


def filter_by_mask(samples, sample_mask):
	if len(samples) != len(sample_mask):
		logTracker.logException('Inconsistent samples size and sample_mask size: ' + str(len(samples)) + ' - ' + str(len(sample_mask)))
	return [samples[i] for i in range(len(samples)) if sample_mask[i] is True]


def filter_by_indices(feed_dict, indices):
	feed_dict = dict(feed_dict)
	for trainable in feed_dict:
		feed_dict[trainable] = [d for i, d in enumerate(feed_dict[trainable]) if i in indices]
	return feed_dict


def get_batches(indices, batch_size):
	res = []
	batch = []
	for i in indices:
		batch.append(i)
		if len(batch) >= batch_size:
			res.append(batch)
			batch = []
	if len(batch) > 0:
		res.append(batch)
	return res


