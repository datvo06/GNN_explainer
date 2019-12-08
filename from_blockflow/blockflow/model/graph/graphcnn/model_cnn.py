__author__ = 'Ethan'


import numpy as np
import tensorflow as tf

from graphkv.algorithm.architecture import layers

class ModelCNN:
    def __init__(self, num_vertex_feature, num_class,
        num_edge, is_training, global_step):
        self.num_class = num_class
        self.num_vertex_feature = num_vertex_feature
        self.num_edge = num_edge
        self.current_V = None
        self.current_A = None
        self.network_debug = False
        self.current_pos = None
        self.is_training = is_training
        self.global_step = global_step

        # Build network model
        self.build_network()


    def build_network(self):
        self.create_input()
        # e1, _ = self.make_embedding_layer(200)
        e2, _ = self.make_embedding_layer(128)
        self.make_dropout_layer()

        self.make_graphcnn_layer(128)
        g1 = self.make_dropout_layer()
        # self.current_V = tf.add(g1, e2)
        a1 = self.current_V

        self.make_graphcnn_layer(128)
        g2 = self.make_dropout_layer()
        # self.current_V = tf.add(g2, a1)
        self.current_V = tf.concat([g2, g1], -1)
        a2 = self.current_V

        self.make_graphcnn_layer(128)
        g3 = self.make_dropout_layer()

        self.current_V = tf.concat([g3, g1], -1)
        # self.current_V = tf.add(g3, a1)
        #self.make_embedding_layer(256)
        self.make_embedding_layer(128)
        self.make_self_atten(128, 'atten1')

        #self.make_embedding_layer(64)
        self.make_dropout_layer()
        self.make_embedding_layer(64)
        self.make_dropout_layer()
        self.make_embedding_layer(32)

        self.make_embedding_layer(
            self.num_class, name='final', with_bn=False, with_act_func=False)

        return self.V_in, self.A_in, self.current_V
    
    def create_input(self):
        self.V_in = tf.placeholder(dtype=tf.float32, shape=[None,
                                                            None, self.num_vertex_feature], name='input_vertices')
        self.A_in = tf.placeholder(dtype=tf.float32, shape=[
            None, None, self.num_edge, None], name='input_adj')
        self.current_V = self.V_in
        self.current_A = self.A_in

        # Visual helper
        self.current_pos = None
        self.visual_indices = None

        if self.network_debug:
            size = tf.reduce_sum(self.current_mask, axis=1)
            self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), tf.reduce_max(
                size), tf.reduce_mean(size)], message='Input V Shape, Max size, Avg. Size:')

    def make_batchnorm_layer(self):
        self.current_V = layers.make_bn(
            self.current_V, self.is_training, num_updates=self.global_step)
        return self.current_V

    # Equivalent to 0-hop filter
    def make_embedding_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Embed') as scope:
            self.current_V = layers.make_embedding_layer(
                self.current_V, no_filters)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
        return self.current_V, self.current_A

    def make_dropout_layer(self, keep_prob=0.5, name=None):
        with tf.variable_scope(name, default_name='Dropout') as scope:
            self.current_V = tf.cond(self.is_training, lambda: tf.nn.dropout(
                self.current_V, keep_prob=keep_prob), lambda: (self.current_V))
            return self.current_V

    def make_graphcnn_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Graph-CNN') as scope:
            self.current_V = layers.make_graphcnn_layer(
                self.current_V, self.current_A, no_filters)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(
                    self.current_V, np.arange(len(self.current_V.get_shape())-1))
                self.current_V = tf.Print(self.current_V, [tf.shape(
                    self.current_V), batch_mean, batch_var], message='"%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V

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

        return self.current_V