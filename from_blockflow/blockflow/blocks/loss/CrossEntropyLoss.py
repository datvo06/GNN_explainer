import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from chi_lib.library import *
from blocks.Block import Block


class CrossEntropyLoss(Block):
	def __init__(self, logits_block, label_block, normalize=False, name=None):
		Block.__init__(self, [logits_block, label_block], name=name)
		self.logits_block = logits_block
		self.label_block = label_block
		self.normalize = normalize
		
	def implement(self):
		logits, _ = self.logits_block.get()['features']
		temp = self.label_block.get()
		labels, _ = temp['labels']
		labels = tf.cast(labels, tf.int64)
		
		loss_weights = 1.0
		if self.normalize:
			n_classes = logits.get_shape()[-1].value
			labels_onehot = tf.one_hot(labels, depth=n_classes, axis=-1)
			old_shape = tf.shape(labels_onehot)
			labels_onehot = tf.reshape(labels_onehot, (-1, old_shape[-1]))

			classes_weights = tf.reduce_sum(labels_onehot, axis=0)
			classes_weights = tf.reshape(classes_weights, (-1, 1))
			classes_weights = tf.div(1.0, (classes_weights + 1))

			loss_weights = tf.matmul(labels_onehot, classes_weights)
			loss_weights = tf.reshape(loss_weights, tf.stack(old_shape[:-1]))
			#loss_weights = tf.expand_dims(loss_weights, axis=-1)
		
		#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.current_V, labels=value_labels)
		labels = tf.squeeze(labels, axis=-1)

		cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels, weights=loss_weights)
		cross_entropy = tf.reduce_mean(cross_entropy) 

		correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 2), labels), tf.float32)
		accuracy = tf.reduce_mean(correct_prediction)
		return {
			'loss' : (cross_entropy, None),
			'accuracy' : (accuracy, None)
		}


