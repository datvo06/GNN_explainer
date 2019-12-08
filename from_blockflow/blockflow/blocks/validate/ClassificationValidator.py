import os, sys, json
import numpy as np
import tensorflow as tf

sys.path.append('../datahub/') 
from sklearn.metrics import confusion_matrix

from chi_lib.ProgressBar import ProgressBar
from chi_lib.library import *
from blocks.Block import Block
from blocks.loss.CrossEntropyLoss import CrossEntropyLoss
from utils.utils import * 


class ClassificationValidator(Block):
	def __init__(self, logits_block, labels_block, name=None):
		loss_block = CrossEntropyLoss(logits_block, labels_block, normalize=False, name=name)
		Block.__init__(self, [logits_block, labels_block, loss_block], name=name)
		self.logits_block = logits_block
		self.labels_block = labels_block
		self.loss_block = loss_block
		
	def implement(self):
		#self.accuracy, _ = self.loss_block.get()['accuracy']
		return {}

	def execute(self):
		self.accuracy, _ = self.loss_block.get()['accuracy']
		self.losses, _ = self.loss_block.get()['loss']
		self.logits, _ = self.logits_block.get()['features']
		self.labels, _ = self.labels_block.get()['labels']

		n_classes = self.logits.get_shape()[-1].value
		classes = list(range(n_classes))
		
		information_dict = self.get()
		feed_dict = self.get_masked_feed()

		n_samples = None
		for trainable in feed_dict:
			if n_samples is None:
				n_samples = len(feed_dict[trainable])
			if len(feed_dict[trainable]) != n_samples:
				logTracker.logException('Inconsistent sample size: ' + str(len(feed_dict[trainable])) + ' - ' + str(n_samples))

		logTracker.log('Validating ' + str(n_samples) + ' samples')		
		session = self.get_tf_session()

		res = {
			'validation_losses' : [],
			'validation_accuracy' : [],
			'ground_truth' : [],
			'prediction' : [],
		}

		is_training = None
		if 'is_training' in information_dict:
			is_training, _ = information_dict['is_training']

		progress = ProgressBar('Validating', n_samples)
		for i in range(n_samples):
			temp_feed_dict = filter_by_indices(feed_dict, [i])
			for d in temp_feed_dict:
				temp_feed_dict[d] = np.stack(temp_feed_dict[d], axis=0)
				
			if not is_training is None:
				temp_feed_dict[is_training] = False

			val_losses, val_accuracy, val_logits, val_labels = session.run([self.losses, self.accuracy, self.logits, self.labels], feed_dict=temp_feed_dict)
			
			res['validation_losses'].append(val_losses)
			res['validation_accuracy'].append(val_accuracy)
			
			prediction = np.argmax(val_logits, axis=-1).squeeze().tolist()
			val_labels = np.squeeze(val_labels[0]).tolist()

			res['ground_truth'].extend(val_labels)
			res['prediction'].extend(prediction)
			progress.increase()
		progress.done()

		cf = confusion_matrix(res['ground_truth'], res['prediction'], labels=classes)
		res['confusion_matrix'] = cf
		res['precision'] = precision_macro_average(cf)
		res['recall'] = recall_macro_average(cf)

		log_mess = [
			'Avg validate loss: {}, avg validate accuracy: {}'.format(np.round(np.mean(res['validation_losses']), 5), np.round(np.mean(res['validation_accuracy']), 5)),
			'Precision: {}, Recall: {}'.format(np.round(res['precision'], 5), np.round(res['recall'], 5)),
			'Confusion matrix:',
			str(res['confusion_matrix']),
			'Confusion matrix shape: ' + str(res['confusion_matrix'].shape)
		]
		res['log'] = mergeList(log_mess, '\n')
		return res


def precision(label, confusion_matrix):
	col = confusion_matrix[:, label]
	return confusion_matrix[label, label] / col.sum()


def recall(label, confusion_matrix):
	row = confusion_matrix[label, :]
	return confusion_matrix[label, label] / row.sum()


def precision_macro_average(confusion_matrix):
	rows, columns = confusion_matrix.shape
	sum_of_precisions = 0
	for label in range(rows):
		sum_of_precisions += precision(label, confusion_matrix)
	if rows < 1e-4:
		return np.nan
	return sum_of_precisions / rows


def recall_macro_average(confusion_matrix):
	rows, columns = confusion_matrix.shape
	sum_of_recalls = 0
	for label in range(columns):
		sum_of_recalls += recall(label, confusion_matrix)
	if columns < 1e-4:
		return np.nan
	return sum_of_recalls / columns