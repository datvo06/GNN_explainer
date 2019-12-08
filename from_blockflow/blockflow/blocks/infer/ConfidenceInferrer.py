import os, sys, json
import numpy as np
import tensorflow as tf
from pprint import pprint
sys.path.append('../datahub/')

from utils.utils import load_frozen_graph
from chi_lib.library import *
from utils.utils import *
from chi_lib.ProgressBar import ProgressBar
from blocks.Block import Block


def print_log(obj, obj_info=""):
	if type(obj) is np.ndarray:
		print("\n\n\n" + obj_info + " = \n%r,\n\nWith shape = %s\n\n" % (obj, obj.shape))
	elif type(obj) is list:
		print("\n\n" + obj_info + " : %r" % obj)
		print("list of Len: %d" % len(obj))


class ConfidenceInferrer(Block):
	def __init__(self, logits_block, input_block, labels_block=None, name=None):
		if labels_block:
			Block.__init__(self, [logits_block, labels_block], name=name)
		else:
			Block.__init__(self, [logits_block], name=name)
		# Block.__init__(self, [logits_block, input_block, labels_block], name=name)
		self.logits_block = logits_block
		self.input_block = input_block
		self.labels_block = labels_block

	def implement(self):
		return {}

	def execute(self):
		self.logits, _ = self.logits_block.get()['features']
		self.weights , _ = self.logits_block.get()['weights']
		self.inputs, _ = self.input_block.get()['features']
		self.labels, _ = self.labels_block.get()['labels']
		_, self.f_paths = self.input_block.get()['json_file_paths']

		n_classes = self.logits.get_shape()[-1].value
		classes = list(range(n_classes))
		print("\n\n--------- Weights ---------:",
			  *self.weights,
			  sep="\n")

		print("\n\nSamples:",
			  *[os.path.basename(p) for p in self.f_paths],
			  sep="\n")

		mask = self.get_mask()
		self.f_paths = filter_by_mask(self.f_paths, mask)

		information_dict = self.get()
		feed_dict = self.get_masked_feed()

		n_samples = None
		for trainable in feed_dict:
			if n_samples is None:
				n_samples = len(feed_dict[trainable])
			if len(feed_dict[trainable]) != n_samples:
				logTracker.logException(
					'Inconsistent sample size: ' + str(len(feed_dict[trainable])) + ' - ' + str(n_samples))

		logTracker.log('Inferring ' + str(n_samples) + ' samples')
		session = self.get_tf_session()

		is_training = None
		if 'is_training' in information_dict:
			is_training, _ = information_dict['is_training']


		######## Work here ########

		max_logits = tf.reduce_max(self.logits, axis=-1)  # .squeeze()
		lines_identity = tf.placeholder(dtype=tf.float32, name="lines_identity")

		# with tf.device("/job:localhost/replica:0/task:0/device:XLA_GPU:0"):
		temp = tf.gradients(ys=max_logits, xs=self.weights, grad_ys=lines_identity)

		###########################

		# confidences = []
		predictions = []
		embeddings_all = []
		logits_all = []
		labels_all = []
		predictions_correct = []

		grad_maxlogit_weights_all = []
		grad_maxlogit_bow_all = []

		progress = ProgressBar('Inferring', n_samples)
		for i in range(n_samples):
			temp_feed_dict = filter_by_indices(feed_dict, [i])
			for d in temp_feed_dict:
				temp_feed_dict[d] = np.stack(temp_feed_dict[d], axis=0)

			if not is_training is None:
				temp_feed_dict[is_training] = False

			############ Work here <3 <3 ############

			# -------------- Don't change -----------------

			embedding_input, output_vals, val_labels = \
				session.run([self.inputs, self.logits, self.labels], feed_dict=temp_feed_dict)

			prediction = np.argmax(output_vals, axis=-1).squeeze()
			val_labels = val_labels.squeeze()
			pred_correct = (prediction == val_labels)

			logits_all.append(output_vals)
			embeddings_all.append(embedding_input)
			labels_all.append(val_labels)
			predictions.append(prediction)
			predictions_correct.append(pred_correct)

			print("\n\nLine Cuts in this sample: %d" % len(prediction))
			# print_log(prediction, "Class Prediction")
			# print_log(val_labels, "Class Ground Truth")
			# print_log(pred_correct, "Predict correct")
            #
			# print_log(embedding_input, "Embedding input (Bag of words)")
			# print_log(output_vals, "Logits")  # Shape: (batch, textlines, classes)

			# max_logits = tf.reduce_max(self.logits, axis=-1)  # .squeeze()
			# print("max_logits: ", max_logits)
			# max_logits Shape: (batch, textlines)

			# ------------------------------------------------

			# ####### Experiment #######

			# self.weights is a list with 14 tf weight variables.
			# weights = session.run(self.weights, feed_dict=temp_feed_dict)
			# print_log(weights[1], "Gcn0_weights:")

			# temp = tf.gradients(ys=max_logits[:, 7], xs=self.weights)
			# temp = [tf.gradients(ys=max_logits[:, i], xs=self.weights) for i in range(len(prediction))]
			# temp = session.run(temp, feed_dict=temp_feed_dict)
			# print_log(temp, "Many gradients for text line 7")
			# print(type(temp), temp, temp.shape)
			# print(type(temp), [t.shape for t in temp[0]])


			# All kind of Gradients of max_logits w.r.t. weights

			# temp = tf.gradients(ys=max_logits, xs=self.weights)
			# grad_maxlogit_weights = session.run(temp, feed_dict=temp_feed_dict)
			# grad_maxlogit_weights_all.append(grad_maxlogit_weights)
			# print_log(grad_maxlogit_weights, "Gradient of Maxlogit w.r.t. all Weights")


			# for grad in grad_maxlogit_weights:
			# 	print(grad.shape)
			# print(*self.weights, sep="\n")

			grad_maxlogit_weights_per_sample = []
			identity_matrix = np.eye(len(prediction), dtype=int)
			for textline_idx in range(len(prediction)):
				temp_feed_dict[lines_identity.name] = identity_matrix[textline_idx]
				grad_maxlogit_weights = session.run(temp, feed_dict=temp_feed_dict)
				grad_maxlogit_weights_per_sample.append(grad_maxlogit_weights)
				# print("textline_idx = ", textline_idx)
				# print("Idd = ", identity_matrix[textline_idx])
				# print_log(grad_maxlogit_weights[2], "grad_maxlogit_weights: ")

			grad_maxlogit_weights_all.append(grad_maxlogit_weights_per_sample)

			'''
			temp = tf.gradients(ys=max_logits, xs=self.inputs)
			grad_maxlogit_bow = session.run(temp, feed_dict=temp_feed_dict)[0]
			grad_maxlogit_bow_all.append(grad_maxlogit_bow)
			print_log(grad_maxlogit_bow, "Gradient of Max logit w.r.t. Bag of words")
			# TODO: the tf.gradients CANNOT compute jacobian matrix w.r.t. different ys.
			
			# "TypeError: Fetch argument None has invalid type <class 'NoneType'>" occurs,
			# while slicing to self.inputs[:, 7, :]
			temp = tf.gradients(ys=max_logits[:, 7], xs=self.inputs)
			temp = session.run(temp, feed_dict=temp_feed_dict)[0]
			print_log(temp, "Gradient of Max logit w.r.t. Bag of words, textline 7")
			# print(np.sum(grad_maxlogit_bow == temp))
			'''

			# ###### some confidence score here ###### #
			#
			# exit()

			if sum(pred_correct.__invert__()):
				print("\nPred wrong happens in:\nTestline {} of\n {}\n\n"
					  .format(np.where(pred_correct==False), self.f_paths[i]))

			progress.increase()
		progress.done()

		# if len(confidences) != len(self.f_paths):
		# 	logTracker.logException('Inconsistent confidence score length: {} - {}'.format(len(confidences), len(self.f_paths)))
		# confidences = list(zip(self.f_paths, confidences, predictions_correct))

		# return {
		# 	'data path': self.f_paths,
		# 	'bag of words': embeddings_all,
		# 	'logits': logits_all,
		# 	'predictions': predictions,
		# 	'label': labels_all,
		# 	'weights': session.run(self.weights, feed_dict=temp_feed_dict),
		# 	'weights_name': [w.name for w in self.weights],
		# 	'grad_maxlogit_weights': grad_maxlogit_weights_all,
		# 	'grad_maxlogit_bow': grad_maxlogit_bow_all
		# }
		del grad_maxlogit_weights_all
		del labels_all
		del embeddings_all

		return {'data path': self.f_paths}



