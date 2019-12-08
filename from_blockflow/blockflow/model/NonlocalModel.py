
__author__ = 'Chi.Tran'


import tensorflow as tf
import numpy as np
import math

from model.graph.graphcnn import layers
from sklearn.metrics import confusion_matrix
from .EthanGraphCNNModel import EthanGraphCNNModel


class NonlocalModel(EthanGraphCNNModel):
	def __init__(self, bow_features_tensor, coord_features_tensor):
		EthanGraphCNNModel.__init__(self, bow_features_tensor, coord_features_tensor, None)
		
	def build(self):
		self.create_input()
		self.build_non_local_nn()
		self.ethan_gcnn_block('gcnn')


	# ######################## Building block methods ######################## 


	def build_non_local_nn(self):
		temp_V = self.current_V
		# Attend spatial info (last 4 dimensions)
		self.current_V = temp_V[:, :, -4:]
		#self.__non_local_res_gcnn_block('spatial_nl1', 128, 8, 'softmax')
		_, spatial_logits_A = self.__non_local_res_gcnn_block('spatial_nl2', 4, 8, 'softmax', embedding_dropout=False)
		spatial_embedding_V = self.current_V
		spatial_embedding_A = spatial_logits_A

		# Attend semantic info (all except the last 4 dimensions)
		self.current_V = temp_V[:, :, :-4]
		#self.__non_local_res_gcnn_block('semantic_nl1', 128, 8, 'softmax')
		_, semantic_logits_A = self.__non_local_res_gcnn_block('semantic_nl2', 256, 8, 'softmax')
		semantic_embedding_V = self.current_V
		semantic_embedding_A = semantic_logits_A

		# Aggregate info of spatial features and semantic feature
		self.current_V = tf.concat([spatial_embedding_V, semantic_embedding_V], axis=-1) # [n_batchs, n_vertices, n_spatial_features + n_semantic_features]
		# Embedding on the node feature dimension
		self.make_embedding_layer(128, axis=-1, name='aggregated_features') # [n_batchs, n_vertices, n_features]
		
		# Aggregate info of spatial attentions and semantic attentions (we use current_V as a temporary variable to embed current_A)
		temp_V = self.current_V
		self.current_V = tf.concat([spatial_embedding_A, semantic_embedding_A], axis=-2) # [n_batchs, n_vertices, num_spatial_relations + num_semantic_relations, n_vertices]
		# Embedding on the relation stacking dimension
		self.make_embedding_layer(8, axis=-2, with_act_func=False, with_bn=False, name='aggregated_attentions') # [n_batchs, n_vertices, num_relations, n_vertices]
		self.current_A = self.current_V
		self.current_A = tf.nn.softmax(self.current_A)
		self.current_V = temp_V

	def __non_local_res_gcnn_block(self, block_name, gcnn_features, n_relations=8, corr_type='softmax', embedding_dropout=True):
		temp_V = self.current_V
		self.make_embedding_layer(gcnn_features)
		embedding_V = self.current_V
		self.current_V = temp_V

		if embedding_dropout is True:
			self.make_dropout_layer()

		final_A, logits_A = self.make_non_local_block(n_relations, corr_type, block_name + '_nonlocal_1')
		self.current_A = final_A

		self.make_graphcnn_layer(gcnn_features, block_name + '_gcnn_1_1')
		if embedding_dropout is True:
			self.make_dropout_layer()

		self.current_V = tf.concat([self.current_V, embedding_V], axis=-1)
		self.make_embedding_layer(gcnn_features)
		return final_A, logits_A

	def make_non_local_block(self, num_relations, corr_type, name=None, with_bn=True, with_act_func=True):
		with tf.variable_scope(name, default_name='Nonlocal_block') as scope:
			self.current_V, final_A, logits_A = layers.make_non_local_block(self.current_V, num_relations, self.is_training, self.global_step, corr_type)
			if with_bn:
				self.make_batchnorm_layer()
			if with_act_func:
				self.current_V = tf.nn.relu(self.current_V)
		return final_A, logits_A
