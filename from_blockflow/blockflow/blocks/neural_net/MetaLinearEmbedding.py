import os, sys, json
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from model.graph.graphcnn import layers
from blocks.Block import Block


class MetaLinearEmbedding(Block):
	def __init__(self, grad_norm_features_block, name=None):
		Block.__init__(self, [grad_norm_features_block], name=name)
		self.grad_norm_features_block = grad_norm_features_block
		
	def implement(self):
		grad_norm_features, _ = self.grad_norm_features_block.get()['gradients']

		input_dim = len(grad_norm_features)
		grad_norms = tf.stack(grad_norm_features, axis=0)

		w = layers.make_variable(name="weights_meta_classify", shape=(1, input_dim))
		b = layers.make_bias_variable(name="bias_meta_classify", shape=(1, 1))
		y = tf.matmul(w, grad_norms) + b

		score = tf.div(1.0, tf.add(1.0, tf.exp(-y)))

		return {
			'gradients': (grad_norm_features, None),
			'meta_score': (score, None)
		}

'''
if __name__ == '__main__':

	n = 7

	gradients = [tf.placeholder(dtype=tf.float32,
								shape=(1,),
								name="gradient_features") for w in range(n)]

	gradients = tf.stack(gradients, axis=0)

	w = layers.make_variable("weight_meta_classify", (1, n))
	b = layers.make_bias_variable(name="bias_meta_classify", shape=(1, 1))

	y = tf.matmul(w, gradients) + b


	# Debug use:

	feed = {g.name: np.asarray([i]) for i, g in enumerate(gradients)}

	sess = tf.Session()
	sess.run([tf.global_variables_initializer(),
			  tf.local_variables_initializer()])
	g, w_, b_ = sess.run([g_2, w, b], feed_dict=feed)
	y_ = sess.run(y, feed_dict=feed)
'''
