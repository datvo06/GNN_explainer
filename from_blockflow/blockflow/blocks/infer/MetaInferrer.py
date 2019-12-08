import os, sys, json
import numpy as np
import tensorflow as tf
from pprint import pprint
sys.path.append('../datahub/')

from chi_lib.library import *
from utils.utils import *
from chi_lib.ProgressBar import ProgressBar
from blocks.Block import Block
from blocks.loss.CrossEntropyLoss import CrossEntropyLoss
from model.graph.graphcnn.layers import make_softmax_layer

# TODO: write different norm functions and load them in.


def print_log(obj, obj_info="", new_line=False):
	if type(obj) is np.ndarray:
		print("\n\n\n" + obj_info + " = \n%r,\n\nWith shape = %s\n\n" % (obj, obj.shape))
	elif type(obj) is list:
		if new_line:
			print("\n\n%s:" % obj_info, *obj, sep="\n")
		else:
			print("\n\n" + obj_info + " : %r" % obj)
		print("list of Len: %d" % len(obj))
	else:
		print("\n%s:" % obj_info, obj, type(obj), sep="\n")


def gradient_norm(text_grad):
	"""
	:param text_grad:
			:type: np.ndarray
			A gradient w.r.t. one of the weights.
	:return:
			:type: np.float !?
			:type: np.ndarray with shape (1,).
			# TODO: update shape when we need batch.
	"""
	# Mask the 0
	weight_dim = text_grad.shape[0] * text_grad.shape[1]
	text_grad = text_grad[text_grad != 0]
	# normalize
	text_grad = (text_grad * weight_dim) / len(text_grad)

	norm_function = np.linalg.norm
	grad_norm = norm_function(text_grad)

	return np.asarray(grad_norm).reshape((1,))


def CrossEntropy(yHat, y):
	"""
	:param yHat:
		:type: tf tensor.
		Defined in MetaLinearEmbedding.
		Probability of model binary prediction.
		After session.run(yHat), it shall be <class 'numpy.ndarray'>, with shape = (1, 1).
	:param y:
		:type:
		True / False label of binary problem.
	:return: CrossEntropy loss tensorflow node.
	"""
	# if y == 1: return -tf.log(yHat)
	# else: return -tf.log(1 - yHat)

	# TODO: Sometimes return nan when running the tensor???
	# return (1-y)* -tf.log(1 - yHat) + y * -tf.log(yHat)

	return -((1-y)* tf.log(1 - yHat + 1e-10) + y * tf.log(yHat))


class MetaInferrer(Block):
	def __init__(self, input_block, labels_block, logits_block, confidence_score_block, name=None):

		# input_block for (Bag of words) has already been init before.
		Block.__init__(self, [labels_block, logits_block, confidence_score_block], name=name)

		self.input_block = input_block
		self.labels_block = labels_block
		self.logits_block = logits_block
		self.score_block = confidence_score_block

		# self.learning_rate = 0.001
		self.learning_rate = 0.01
		self.is_initialized = False


	def implement(self):
		self.get_obj_from_block()

		################################### Work here ###################################
		self.build_other_tensors()
		#################################################################################

		return {}

	def get_obj_from_block(self):
		self.inputs, _ = self.input_block.get()['features']
		self.labels, _ = self.labels_block.get()['labels']

		self.logits, _ = self.logits_block.get()['features']
		self.weights, _ = self.logits_block.get()['weights']

		# self.accuracy, _ = self.loss_block.get()['accuracy']

		self.grad_norm_tensors, _ = self.score_block.get()['gradients']
		self.meta_score, _ = self.score_block.get()['meta_score']

		_, self.f_paths = self.input_block.get()['json_file_paths']

	def build_other_tensors(self):

		print_log(self.weights,
				  obj_info="--------- Weights ---------", new_line=True)
		print_log([os.path.basename(p) for p in self.f_paths],
				  obj_info="--------- Samples ---------", new_line=True)
		print_log(self.grad_norm_tensors,
				  obj_info="--------- Gradient Norm tensors ---------", new_line=True)
		print_log(self.meta_score,
				  obj_info="--------- Meta Score tensors ---------")

		# max_logits for all textlines
		max_logits = tf.reduce_max(self.logits, axis=-1)  # .squeeze()

		self.lines_identity = tf.placeholder(dtype=tf.float32, name="lines_identity")
		self.grad_ml_w = tf.gradients(ys=max_logits, xs=self.weights, grad_ys=self.lines_identity)

		# TODO: Check the shape meaning of self.labels = (1, n_textlines, 1)
		# TODO: [Done] Create Loss Node for linear classifier (True/False classified)
		# Not yet for batch.

		self.predicted_labels = tf.argmax(self.logits, axis=-1)
		predicted_labels = tf.dtypes.cast(self.predicted_labels, tf.int32)

		self.pred_correct = tf.equal(predicted_labels, tf.squeeze(self.labels, axis=-1))

		self.meta_y = tf.placeholder(dtype=tf.float32, name="meta_y")
		self.loss = CrossEntropy(self.meta_score, self.meta_y)

		# TODO: Create Optimizer Node to optimize Loss for linear classifier
		self.train_op = tf.train.AdamOptimizer(
			self.learning_rate).minimize(
			tf.reduce_sum(self.loss))  # , global_step=global_step)

		# TODO: Pseudo loss
		self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=self.logits,
																	labels=tf.squeeze(self.labels, axis=-1),
																	reduction=tf.losses.Reduction.NONE)
		self.pseudo_cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=self.logits,
																		   labels=self.predicted_labels,
																		   reduction=tf.losses.Reduction.NONE)
		self.grad_p_loss_w = tf.gradients(ys=tf.squeeze(self.pseudo_cross_entropy),
										  xs=self.weights,
										  grad_ys=self.lines_identity)

	def debug_per_sample(self, feed_dict):
		session = self.get_tf_session()
		embedding_input, output_vals, val_labels = session.run([self.inputs,
																self.logits,
																self.labels], feed_dict=feed_dict)
		pred, fake_loss, real_loss = session.run([self.predicted_labels,
												  self.pseudo_cross_entropy,
												  self.cross_entropy], feed_dict=feed_dict)
		# loss_ran = session.run(self.loss, feed_dict=feed_dict)

		prediction_np = np.argmax(output_vals, axis=-1).squeeze()
		val_labels = val_labels.squeeze()
		# pred_correct_np = (prediction_np == val_labels)

		# print_log(output_vals, "Logits, output_vals")
		# print_log(embedding_input, "embedding_input Bag of words")
		# print_log(val_labels, "val_labels")
		# print_log(pred, "tf Predicted labels")

		# print_log(fake_loss, "TF Pseudo fake_loss")
		# print_log(real_loss, "TF real_loss")

		# print_log(prediction_np, "Class prediction")
		# print_log(pred_correct_np, "Numpy Model Pred_correct")

		n_textlines = len(prediction_np)


	def debug_per_textline(self, feed_dict, textline_idx):

		# print("Textline number %d" %textline_idx)
		y_truth = feed_dict[self.meta_y.name]
		session = self.get_tf_session()
		norm_ran, score = session.run([self.grad_norm_tensors,
									   self.meta_score], feed_dict=feed_dict)
		# print_log(norm_ran, "Session run gradient_norms", new_line=True)
		# print_log(score, "meta_score")

		grad_p_loss_w = session.run(self.grad_p_loss_w, feed_dict=feed_dict)
		# print_log(grad_p_loss_w[0], "grad_p_loss_w 0")
		# print_log(grad_p_loss_w[1], "grad_p_loss_w 1")
		# print_log(grad_p_loss_w[2], "grad_p_loss_w 2")
		# print_log(grad_p_loss_w[3], "grad_p_loss_w 3")
		# print_log(grad_p_loss_w[4], "grad_p_loss_w 4")
		# print_log(grad_p_loss_w[5], "grad_p_loss_w 5")

		# lines_identity = session.run(self.lines_identity, feed_dict=feed_dict)
		# print_log(lines_identity, "lines_identity")

		# loss_np = (1 - y_truth) * -np.log(1 - score) + y_truth * -np.log(score)
		# print_log(loss_ran, "loss_ran")

		# if -np.log(score) != loss_ran:
		# 	print(textline_idx, y_truth,
		# 		  loss_ran, -np.log(score), loss_np,
		# 		  sep="\n")

		"""
		RuntimeWarning: divide by zero encountered in log
		RuntimeWarning: invalid value encountered in multiply
		loss_np = (1 - y_truth) * -np.log(1 - score) + y_truth * -np.log(score)
		"""


	def execute(self):
		# n_classes = self.logits.get_shape()[-1].value

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
		if self.is_initialized is False:
			logTracker.log('Initialize variables')
			session.run(tf.global_variables_initializer())
			self.is_initialized = True

		is_training = None
		if 'is_training' in information_dict:
			is_training, _ = information_dict['is_training']

		progress = ProgressBar('Inferring', n_samples)
		losses, json_output = [], []
		truth_sample_prediction, false_sample_prediction = [], []
		all_ml_w_gradient_norms, all_meta_y = [], []
		all_pseudo_loss_w_gradient_norms = []
		all_logits, all_loss, all_pseudo_loss = [], [], []
		for i in range(n_samples):
			output_per_sample = {}

			temp_feed_dict = filter_by_indices(feed_dict, [i])
			for d in temp_feed_dict:
				temp_feed_dict[d] = np.stack(temp_feed_dict[d], axis=0)

			if not is_training is None:
				temp_feed_dict[is_training] = False

			############ Work here <3 <3 ############
			predict_ran = session.run(self.pred_correct, feed_dict=temp_feed_dict)
			n_textlines = predict_ran.shape[-1]
			identity_matrix = np.eye(n_textlines, dtype=int)

			output_logits, fake_loss, real_loss = \
				session.run([self.logits, self.pseudo_cross_entropy, self.cross_entropy],
							feed_dict=temp_feed_dict)
			all_logits.append(output_logits)
			all_loss.append(real_loss)
			all_pseudo_loss.append(fake_loss)

			# print_log(predict_ran, "Session run Model Pred_correct")
			# print("\n\nLine Cuts in this sample: %d" % n_textlines)
			self.debug_per_sample(temp_feed_dict)

			sample_meta_score = []
			for textline_idx in range(n_textlines):
				temp_feed_dict[self.lines_identity.name] = identity_matrix[textline_idx]

				# TODO: Compute norm for all grad_pseudo_loss_weights[i]
				grad_p_loss_weights = session.run(self.grad_p_loss_w, feed_dict=temp_feed_dict)
				# print_log(grad_p_loss_weights[0], "Grad_p_loss_weights_ran")
				p_loss_weights_grad_norm = [gradient_norm(g) for g in grad_p_loss_weights]
				# all_pseudo_loss_w_gradient_norms.append()

				# TODO: [Done] Compute norm for all grad_maxlogit_weights[i]
				# grad_maxlogit_weights is a list of length 6 refering 6 weights.
				grad_maxlogit_weights = session.run(self.grad_ml_w, feed_dict=temp_feed_dict)
				maxlogit_weights_grad_norm = [gradient_norm(g) for g in grad_maxlogit_weights]
				# print_log(norm, "gradient_norms", new_line=True)
				# print(type(norm[0]), np.asarray(norm[0]).shape)

				# TODO: [Done] Feed norm into self.grad_norm_tensors.
				gradient_dict = {t.name: n
								 for t, n in zip(self.grad_norm_tensors, maxlogit_weights_grad_norm)}
				gradient_dict[self.meta_y.name] = predict_ran[:, textline_idx]
				# gradient_dict = {**temp_feed_dict, **gradient_dict}
				# self.debug_per_textline(gradient_dict, textline_idx)

				# TODO: [Done] session.run the Optimizer, train the Linear classifier
				_, loss, meta_score = session.run([self.train_op, self.loss, self.meta_score],
															 feed_dict=gradient_dict)
				# print_log(norm, "gradient norm before session run, feed into tensor.", new_line=True)

				# TODO: [Done] append all needed outputs.
				all_pseudo_loss_w_gradient_norms.append(np.stack(p_loss_weights_grad_norm))
				all_ml_w_gradient_norms.append(np.stack(maxlogit_weights_grad_norm))
				all_meta_y.append(predict_ran[:, textline_idx])
				losses.append(loss.squeeze())
				sample_meta_score.append(meta_score.squeeze().tolist())

				if predict_ran[:, textline_idx].__invert__():
					if meta_score.squeeze() <= 0.5:
						false_sample_prediction.append(True)
					else:
						false_sample_prediction.append(False)
					print("Pred wrong in: Textline {}, with Loss: {}, Score: {}"
						  .format(textline_idx, loss.squeeze(), meta_score.squeeze()))
				else:
					if meta_score.squeeze() > 0.5:
						truth_sample_prediction.append(True)
					else:
						truth_sample_prediction.append(False)

			output_per_sample['meta_predict'] = predict_ran.squeeze().tolist()
			output_per_sample['meta_score'] = sample_meta_score

			json_output.append(output_per_sample)

			# print_log(grad_maxlogit_weights[0], obj_info="Gradient of First weight")
			# exit()

			"""
			# if sum(predict_ran.any()):
				print("\nPred wrong happens in:\nTextline {} of\n {}\n\n"
					  .format(np.where(predict_ran==False), self.f_paths[i]))
			"""

			progress.increase()

		# TODO: Consider training with mini batch here ??????????

		assert len(losses)==len(truth_sample_prediction) + len(false_sample_prediction)
		print("\nAverage loss of %d textlines = %f\n\n" %(len(losses), sum(losses)/len(losses)))

		print("\nPrecision for %d True textlines = %f\n\n"
			  %(len(truth_sample_prediction), sum(truth_sample_prediction)/len(truth_sample_prediction)))
		print("Precision for %d False textlines = %f\n\n"
			  %(len(false_sample_prediction), sum(false_sample_prediction)/len(false_sample_prediction)))

		print("\nAverage Meta Accuracy of %d textlines = %f\n\n"
			  %(len(losses), sum(truth_sample_prediction + false_sample_prediction)/len(losses)))
		progress.done()

		# if len(confidences) != len(self.f_paths):
		# 	logTracker.logException('Inconsistent confidence score length: {} - {}'.format(len(confidences), len(self.f_paths)))

		json_output = list(zip(self.f_paths, json_output))
		# log = json.dumps(json_output, indent=4, sort_keys=True)
		return {
			# 'data path': self.f_paths,
			'json_outputs': json_output,
			# 'log': log,  # "\n\nLog prints stuff... \n\n"
			'all_ml-w_gradient_norms': np.concatenate(all_ml_w_gradient_norms,
													  axis=-1),
			'all_pseudo_loss_w_gradient_norms': np.concatenate(all_pseudo_loss_w_gradient_norms,
															   axis=-1),
			'all_meta_y': np.stack(all_meta_y,
								   axis=-1),
			'all_logits': np.concatenate(all_logits,
										 axis=1) ,
			'all_loss': np.concatenate(all_loss,
									   axis=-1),
			'all_pseudo_loss': np.concatenate(all_pseudo_loss,
											  axis=-1)
		}

"""
		return {
			'data path': self.f_paths,
			'meta predict': all_meta_predict,
			'all meta score': all_meta_score
				}
"""
