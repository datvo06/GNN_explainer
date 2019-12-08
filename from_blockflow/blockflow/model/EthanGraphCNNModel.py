__author__ = 'Ethan, Chi.Tran'


import tensorflow as tf
import numpy as np
import math

from model.graph.graphcnn import layers
from sklearn.metrics import confusion_matrix
from random import shuffle


class EthanGraphCNNModel:
	def __init__(self, bow_features_tensor, coord_features_tensor, adj_mat_tensor):
		# tf.reset_default_graph()
		self.bow_features_tensor = bow_features_tensor
		self.coord_features_tensor = coord_features_tensor
		self.adj_mat_tensor = adj_mat_tensor 
		self.is_training = tf.placeholder(tf.bool, name='is_training')
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		self.net_size = 256
		
	def ethan_gcnn_block(self, name):
		self.make_embedding_layer(self.net_size)
		self.make_dropout_layer()

		self.make_graphcnn_layer(self.net_size)
		self.make_dropout_layer()
		g1 = self.current_V
		
		self.make_graphcnn_layer(self.net_size)
		self.make_dropout_layer()
		g2 = self.current_V
		self.current_V = tf.concat([g2, g1], -1)
		
		self.make_graphcnn_layer(self.net_size)
		self.make_dropout_layer()
		g3 = self.current_V

		self.current_V = tf.concat([g3, g1], -1)
		self.make_embedding_layer(self.net_size)
		self.make_self_atten(self.net_size, name + '_atten1')

		self.make_dropout_layer()
		self.make_embedding_layer(int(self.net_size / 2))
		self.make_dropout_layer()
		self.make_embedding_layer(int(self.net_size / 2))

	def create_input(self):
		self.current_V = tf.concat([self.bow_features_tensor, self.coord_features_tensor], axis=-1)
		self.current_A = self.adj_mat_tensor

	def build(self):
		self.create_input()
		self.ethan_gcnn_block('gcnn')


	# ######################## Building block methods ######################## 

	def make_batchnorm_layer(self, axis=-1):
		self.current_V = layers.make_bn(self.current_V, self.is_training, num_updates=self.global_step, axis=axis)
		
	# Equivalent to 0-hop filter
	def make_embedding_layer(self, no_filters, name=None, with_bn=True, with_act_func=True, axis=-1):
		with tf.variable_scope(name, default_name='Embed') as scope:
			self.current_V = layers.make_embedding_layer_on_axis(self.current_V, no_filters, axis=axis)
			if with_bn:
				self.make_batchnorm_layer(axis=axis)
			if with_act_func:
				self.current_V = tf.nn.relu(self.current_V)

	def make_dropout_layer(self, keep_prob=0.5, name=None):
		with tf.variable_scope(name, default_name='Dropout') as scope:
			self.current_V = tf.cond(self.is_training, lambda: tf.nn.dropout(self.current_V, keep_prob=keep_prob), lambda: (self.current_V))
			
	def make_graphcnn_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
		with tf.variable_scope(name, default_name='Graph-CNN') as scope:
			self.current_V = layers.make_graphcnn_layer(self.current_V, self.current_A, no_filters)
			if with_bn:
				self.make_batchnorm_layer()
			if with_act_func:
				self.current_V = tf.nn.relu(self.current_V)
			
	def make_gated_graphcnn_layer(self, num_layers, name=None, with_bn=True, with_act_func=True):
		with tf.variable_scope(name, default_name='Gated-Graph-CNN') as scope:
			_, self.current_V = layers.make_gated_graphcnn_layer(self.current_V, self.current_A, num_layers)
			if with_bn:
				self.make_batchnorm_layer()
			if with_act_func:
				self.current_V = tf.nn.relu(self.current_V)
		
	def make_self_atten(self, no_filters, name=None, reuse=False):
		def hw_flatten(x):
			return tf.reshape(x, shape=[tf.shape(x)[0], -1, x.shape[-1]])
		
		with tf.variable_scope(name, default_name='attention') as scope:
			f = layers.make_embedding_layer(self.current_V, no_filters // 8, name='f')
			g = layers.make_embedding_layer(self.current_V, no_filters // 8, name='g')
			h = layers.make_embedding_layer(self.current_V, no_filters, name='h')

			s = tf.matmul(hw_flatten(g), hw_flatten(
				f), transpose_b=True)

			beta = tf.nn.softmax(s)  # attention map

			o = tf.matmul(beta, hw_flatten(h))
			gamma = tf.get_variable(
				"gamma", [1], initializer=tf.constant_initializer(0.0))

			o = tf.reshape(o, shape=tf.shape(self.current_V))  # [bs, h, w, C]
			self.current_V = gamma * o + self.current_V

	def make_graphcnn_layer_depth(self, no_filters, depth):
		layer_V_in = []
		for i in range(depth):
			# Make skip-connection on each two layers
			if i % 2 == 0 and i != 0:
				self.current_V = tf.concat([layer_V_in[-1], layer_V_in[-2]], -1)

			self.make_graphcnn_layer(no_filters)
			self.make_dropout_layer()
			gi = self.current_V
			layer_V_in.append(gi)

		# Make skip connection on final layer with first layer
		self.current_V = tf.concat([layer_V_in[-1], layer_V_in[0]], -1)
		self.make_embedding_layer(no_filters)
		
	def make_non_local_block(self, num_relations, corr_type, name=None, with_bn=True, with_act_func=True):
		with tf.variable_scope(name, default_name='Nonlocal_block') as scope:
			self.current_V, final_A, logits_A = layers.make_non_local_block(self.current_V, num_relations, self.is_training, self.global_step, corr_type)
			if with_bn:
				self.make_batchnorm_layer()
			if with_act_func:
				self.current_V = tf.nn.relu(self.current_V)
		return final_A, logits_A
