import os, sys
from .helper import *
import tensorflow as tf
import numpy as np
import math
from . import ops
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib import rnn

def make_variable(name, shape, initializer=tf.truncated_normal_initializer(), regularizer=None):
	dtype = tf.float32
	var = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=dtype)
	return var

def make_bias_variable(name, shape):
	dtype = tf.float32
	var = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1), dtype=dtype)
	return var

def make_variable_with_weight_decay(name, shape, stddev=0.01, wd=0.0005):
	dtype = tf.float32
	regularizer = None
	if wd is not None and wd > 1e-7:
		def regularizer(var):
			return tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
	var = make_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev), regularizer=regularizer)
	return var

def get_permutation_to_last(input_shape, axis):
	if axis < 0:
		axis = len(input_shape) + axis

	indices = range(len(input_shape))

	# Move the target dimension to the last dimension
	swapping_permutation = [i for i in indices if i != axis] + [axis]

	# Move the target dimension back to the original position
	original_permulation = [i for i in indices]
	last_idx = original_permulation.pop(-1)
	original_permulation.insert(axis, last_idx)

	return swapping_permutation, original_permulation

def make_bn(V, phase, axis=-1, epsilon=0.001, mask=None, num_updates=None, name=None):
	default_decay = 0.9
	with tf.variable_scope(name, default_name='BatchNorm') as scope:
		swapping_permutation, original_permulation = get_permutation_to_last(V.get_shape(), axis=axis)
		# Move the target dimension to the last dimension
		V = tf.transpose(V, perm=swapping_permutation)	

		# Start the batch normalization on the last dimension
		axis = -1
		input_size = V.get_shape()[axis].value
		if axis < 0:
			axis = len(V.get_shape()) + axis
		
		axis_arr = [i for i in range(len(V.get_shape())) if i != axis]
		if mask == None:
			batch_mean, batch_var = tf.nn.moments(V, axis_arr)
		else:
			batch_mean, batch_var = tf.nn.weighted_moments(V, axis_arr, mask)

		gamma = make_variable('gamma', input_size, initializer=tf.constant_initializer(1))
		beta = make_bias_variable('bias', input_size)
		ema = tf.train.ExponentialMovingAverage(decay=default_decay, num_updates=num_updates)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)
		mean, var = tf.cond(phase, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

		V = tf.nn.batch_normalization(V, mean, var, beta, gamma, 1e-3)

		# Move the target dimension back to the original position
		return tf.transpose(V, perm=original_permulation)

def batch_mat_mult(A, B):
	A_shape = tf.shape(A)
	A_reshape = tf.reshape(A, [-1, A_shape[-1]])

	# So the Tensor has known dimensions
	if B.get_shape()[1] == None:
		axis_2 = -1
	else:
		axis_2 = B.get_shape()[1]
	result = tf.matmul(A_reshape, B)
	result = tf.reshape(result, tf.stack([A_shape[0], A_shape[1], axis_2]))
	return result

def make_softmax_layer(V, axis=1, name=None):
	with tf.variable_scope(name, default_name='Softmax') as scope:
		max_value = tf.reduce_max(V, axis=axis, keep_dims=True)
		exp = tf.exp(tf.subtract(V, max_value))
		prob = tf.div(exp, tf.reduce_sum(exp, axis=axis, keep_dims=True))
		return prob

def make_graphcnn_layer(V, A, no_filters, name=None):
	with tf.variable_scope(name, default_name='Graph-CNN') as scope:
		no_A = A.get_shape()[2].value
		no_features = V.get_shape()[2].value

		W = make_variable_with_weight_decay('weights', [no_features*no_A, no_filters], stddev=math.sqrt(1.0/(no_features*(no_A+1)*1.0)))
		W_I = make_variable_with_weight_decay('weights_I', [no_features, no_filters], stddev=math.sqrt(1.0/(no_features*(no_A+1)*1.0)))
		b = make_bias_variable('bias', [no_filters])

		n = ops.GraphConvolution(V, A)
		A_shape = tf.shape(A)
		n = tf.reshape(n, [-1, A_shape[1], no_A*no_features])
		result = batch_mat_mult(n, W) + batch_mat_mult(V, W_I) + b

		return result

def make_gated_graphcnn_layer(V, A, num_layers, name=None):
	with tf.variable_scope(name, default_name='Gated-Graph-CNN') as scope:
		no_A = A.get_shape()[2].value
		no_features = V.get_shape()[2].value
		gru_cell = tf.keras.layers.GRUCell(no_features)

		W = make_variable_with_weight_decay('weights', [no_features*no_A, no_features], stddev=math.sqrt(1.0/(no_features*(no_A+1)*1.0)))
		b = make_bias_variable('bias', [no_features])
		
		#sub_V = batch_mat_mult(V, W) + b
		for i in range(0, num_layers):
			with tf.variable_scope(name, default_name='Gated-Graph-CNN-Iter' + str(i)) as scope:
				n = ops.GraphConvolution(V, A)
				A_shape = tf.shape(A)
				n = tf.reshape(n, [-1, A_shape[1], no_A*no_features])   # [None, None, no_A * no_features]
				neighbor_info = batch_mat_mult(n, W) + b # Shape ~ [batch_size, no_vertices, 128]
				flatten = tf.reshape(neighbor_info, [-1, no_features])
				
				# Add the sequence length dimension
				flatten = tf.expand_dims(flatten, axis=1)
				gru_input_seq = tf.unstack(flatten, axis=1)
				
				V_shape = tf.shape(V)
				flatten_V = tf.reshape(V, [-1, no_features])
				results, states = rnn.static_rnn(gru_cell, gru_input_seq, initial_state=flatten_V)
				V = tf.reshape(states, V_shape)
				
		#print(states)
		return results, V

def make_graph_embed_pooling(V, A, no_vertices=1, mask=None, name=None):
	with tf.variable_scope(name, default_name='GraphEmbedPooling') as scope:
		factors = make_embedding_layer(V, no_vertices, name='Factors')

		if mask is not None:
			factors = tf.multiply(factors, mask)
		factors = make_softmax_layer(factors)

		result = tf.matmul(factors, V, transpose_a=True)

		if no_vertices == 1:
			no_features = V.get_shape()[2].value
			return tf.reshape(result, [-1, no_features]), A

		result_A = tf.reshape(A, (tf.shape(A)[0], -1, tf.shape(A)[-1]))
		result_A = tf.matmul(result_A, factors)
		result_A = tf.reshape(result_A, (tf.shape(A)[0], tf.shape(A)[-1], -1))
		result_A = tf.matmul(factors, result_A, transpose_a=True)
		result_A = tf.reshape(result_A, (tf.shape(A)[0], no_vertices, A.get_shape()[2].value, no_vertices))

		return result, result_A, factors

def make_embedding_layer(V, no_filters, name=None):
	with tf.variable_scope(name, default_name='Embed') as scope:
		no_features = V.get_shape()[-1].value
		W = make_variable_with_weight_decay('weights', [no_features, no_filters], stddev=1.0/math.sqrt(no_features))
		b = make_bias_variable('bias', [no_filters])
		V_reshape = tf.reshape(V, (-1, no_features))
		s = tf.slice(tf.shape(V), [0], [len(V.get_shape())-1])
		s = tf.concat([s, tf.stack([no_filters])], 0)
		result = tf.reshape(tf.matmul(V_reshape, W) + b, s)
		return result


def make_embedding_layer_on_axis(V, no_filters, axis, name=None):
	swapping_permutation, original_permulation = get_permutation_to_last(V.get_shape(), axis=axis)

	# Move the target dimension to the last dimension
	V = tf.transpose(V, perm=swapping_permutation)
	V = make_embedding_layer(V, no_filters, name=name)

	# Move the target dimension back to the original position
	return tf.transpose(V, perm=original_permulation)


def make_self_atten(V):
	seq_fts = tf.layers.conv1d(V, 1, 1, use_bias=False)
	f_1 = tf.layers.conv1d(seq_fts, 1, 1)
	f_2 = tf.layers.conv1d(seq_fts, 1, 1)
	logits = f_1 + tf.transpose(f_2, [0, 2, 1])
	coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
	vals = tf.matmul(coefs, seq_fts)
	ret = tf.contrib.layers.bias_add(vals)
	return tf.nn.elu(ret)


def make_non_local_block(V, num_relations, training, global_step, corr_type, name=None):
	with tf.variable_scope(name, default_name='NonLocalBlock') as scope:
		V_shape = V.get_shape() # [n_batchs, n_vertices, n_features]
		V_shape_org = tf.shape(V)
		no_V = V_shape[-2].value
		no_features = V_shape[-1].value
		inter_features = int(no_features / 2)

		# Unfold the vertices
		V = tf.reshape(V, (-1, no_features)) # [n_batchs * n_vertices, n_features]

		# We should use tf.transpose to switch the dimension sizes of the tensor, to keep the logic. For instance:
		'''
			Suppose that we have tensor A with shape [3, 5, 4, 6], where 3 is the batch size, 5 is the number of vertices, 4 is the number of adj matrices (relations) and 6 is the size of vertex feature vector.
			
			Now we want to transform A into shape of [3, 4, 5, 6] for latter computing facilitation, instead of using tf.reshape(A, [3, 4, 5, 6]) which will return 
			an incorrect transformation (although it does face any error, and the computational graph is also correct), we should use tf.transpose(A, perm=[0, 2, 1, 3]) which will switch "positions" of the 2nd and the 1st dimensions. 
			=> Explain: 
				- tf.reshape will flatten the tensor A first, and then grouping the elements with group size from left to right (in our case, it went from 6, 5, 4, and 3) to get the new tensor with shape [3, 4, 5, 6]

				- tf.transpose will consider A as 3 subtensors, call Bi, each with shape [5, 4, 6]. Each Bi is treated as a matrix of shape [5, 4], each element of the matrix consists of 6 scalars. Then, it simply transpose that matrix into [4, 5] shape, with element of shape [6] . Finally, the function performs the procedure for all 3 subtensors of shape [5, 4, 6]. Finally, we get the output tensor with shape [3, 4, 5, 6] .

			Example: 
				>>> B
				array([[[[ 0,  1,  2],
						 [ 3,  4,  5],
						 [ 6,  7,  8],
						 [ 9, 10, 11]],
						[[12, 13, 14],
						 [15, 16, 17],
						 [18, 19, 20],
						 [21, 22, 23]]],
					   [[[24, 25, 26],
						 [27, 28, 29],
						 [30, 31, 32],
						 [33, 34, 35]],
						[[36, 37, 38],
						 [39, 40, 41],
						 [42, 43, 44],
						 [45, 46, 47]]]])
				>>> 
				>>> np.reshape(B, [2, 4, 2, 3])
				array([[[[ 0,  1,  2],
						 [ 3,  4,  5]],
						[[ 6,  7,  8],
						 [ 9, 10, 11]],
						[[12, 13, 14],
						 [15, 16, 17]],
						[[18, 19, 20],
						 [21, 22, 23]]],
					   [[[24, 25, 26],
						 [27, 28, 29]],
						[[30, 31, 32],
						 [33, 34, 35]],
						[[36, 37, 38],
						 [39, 40, 41]],
						[[42, 43, 44],
						 [45, 46, 47]]]])
				>>> 
				>>> np.transpose(B, [0, 2, 1, 3])
				array([[[[ 0,  1,  2],
						 [12, 13, 14]],
						[[ 3,  4,  5],
						 [15, 16, 17]],
						[[ 6,  7,  8],
						 [18, 19, 20]],
						[[ 9, 10, 11],
						 [21, 22, 23]]],
					   [[[24, 25, 26],
						 [36, 37, 38]],
						[[27, 28, 29],
						 [39, 40, 41]],
						[[30, 31, 32],
						 [42, 43, 44]],
						[[33, 34, 35],
						 [45, 46, 47]]]])

		'''

		# Compute the 'theta' activation
		W = make_variable_with_weight_decay('weights_theta', [no_features, num_relations * inter_features], stddev=1.0/math.sqrt(no_features))
		b = make_bias_variable('bias_theta', [num_relations * inter_features])
		V_theta = tf.matmul(V, W) + b # [n_batchs * n_vertices, num_relations * inter_features]
		# After flattening into a lower rank tensor, in order to keep the correct order of the original tensor, we have to reshape it back to the original tensor shape (and keeping the order of the original dimensions). 
		V_theta = tf.reshape(V_theta, tf.stack([-1, V_shape_org[-2], num_relations, inter_features]))
		V_theta = tf.transpose(V_theta, perm=[0, 2, 1, 3])
		#V_theta = make_bn(V_theta, training, num_updates=global_step)

		# Compute the 'phi' activation
		W = make_variable_with_weight_decay('weights_phi', [no_features, num_relations * inter_features], stddev=1.0/math.sqrt(no_features))
		b = make_bias_variable('bias_phi', [num_relations * inter_features])
		V_phi = tf.matmul(V, W) + b
		V_phi = tf.reshape(V_phi, tf.stack([-1, V_shape_org[-2], num_relations, inter_features]))
		V_phi = tf.transpose(V_phi, perm=[0, 2, 1, 3])

		# Compute the activation for each vertex
		W = make_variable_with_weight_decay('weights_g', [no_features, num_relations * inter_features], stddev=1.0/math.sqrt(no_features))
		b = make_bias_variable('bias_g', [num_relations * inter_features])
		V_g = tf.matmul(V, W) + b
		V_g = tf.reshape(V_g, tf.stack([-1, V_shape_org[-2], num_relations, inter_features]))
		V_g = tf.transpose(V_g, perm=[0, 2, 1, 3])
		
		# Compute the pairwise relationship matrix and then compute the weighted sum of all activation of vertices
		f_adj = tf.matmul(V_theta, V_phi, transpose_b=True) # [n_batchs, num_relations, n_vertices, n_vertices]
		raw_adj = f_adj

		if corr_type == 'softmax':
			f_adj = tf.nn.softmax(f_adj) # It performs softmax on the last dimension
		elif corr_type == 'sigmoid':
			f_adj = tf.nn.sigmoid(f_adj) # It performs sigmoid on the last dimension
		elif corr_type == 'relu':
			f_adj = tf.nn.relu(f_adj) # It performs relu on the last dimension 
		elif corr_type == 'tanh':
			f_adj = tf.nn.tanh(f_adj) # It performs tanh on the last dimension        
		else:
			raise Exception('Not found corr type: ' + str(corr_type))
		# elif corr_type == 'leaky_relu':
		#     f_adj = tf.nn.leaky_relu(f_adj) # It performs leaky_relu on the last dimension    

		y = tf.matmul(f_adj, V_g) # [n_batchs, num_relations, n_vertices, n_inter_features]
		# Add one activation before reforming
		y = tf.nn.relu(y)

		# This step is to convert the feature size into the original size
		W = make_variable_with_weight_decay('weights_reform_origin', [inter_features, no_features], stddev=1.0/math.sqrt(inter_features))
		b = make_bias_variable('bias_reform_origin', [no_features])
		y = tf.reshape(y, (-1, inter_features)) # [n_batchs * n_vertices * num_relations, n_inter_features]
		y = tf.matmul(y, W) + b # [n_batchs * n_vertices * num_relations, n_features]
		y = tf.reshape(y, tf.stack([-1, V_shape_org[-2], num_relations, no_features])) # [n_batchs, n_vertices, num_relations, n_features]

		# Add one activation before aggregation
		y = tf.nn.relu(y)
		
		# This step is to merge all the adjacency matrices into one weights sum adjacency matrix
		y = tf.transpose(y, perm=[0, 1, 3, 2]) # [n_batchs, n_vertices, n_features, num_relations]
		W = make_variable_with_weight_decay('weights_merge_adjs', [num_relations, 1], stddev=1.0/math.sqrt(num_relations))
		b = make_bias_variable('bias_merge_adjs', [1])
		y = tf.reshape(y, (-1, num_relations)) # [n_batchs * n_vertices * n_features, num_relations]
		y = tf.matmul(y, W) + b # [n_batchs * n_vertices * n_features, 1]
		y = tf.squeeze(y, axis=[1]) # Remove the last dimension of size 1
		y = tf.reshape(y, tf.stack([-1, V_shape_org[-2], no_features])) # [n_batchs, n_vertices, n_features]

		# Add the last activation
		y = tf.nn.relu(y)
			
		# Transform the input V back into the original shape
		V = tf.reshape(V, tf.stack([-1, V_shape_org[-2], no_features])) # [n_batchs, n_vertices, n_features]    

		# Residual layer
		z = y + V # [n_batchs, n_vertices, n_features]
		
		f_adj = tf.transpose(f_adj, perm=[0, 2, 1, 3]) # [n_batchs, n_vertices, num_relations, n_vertices]
		raw_adj = tf.transpose(raw_adj, perm=[0, 2, 1, 3]) # [n_batchs, n_vertices, num_relations, n_vertices]
		return z, f_adj, raw_adj


		
