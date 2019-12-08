import os, argparse, sys
import tensorflow as tf
sys.path.append('../datahub/') 
from chi_lib.library import *
		
# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 
# Ref: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc


def fix_batch_norm_problem(graph_def):
	# https://github.com/davidsandberg/facenet/pull/172/files
	# In order to avoid the ValueError when loading a frozen graph
	for node in graph_def.node:            
		if node.op == 'RefSwitch':
			node.op = 'Switch'
			for index in range(len(node.input)):
				if 'moving_' in node.input[index]:
					node.input[index] = node.input[index] + '/read'
		elif node.op == 'AssignSub':
			node.op = 'Sub'
			if 'use_locking' in node.attr: 
				del node.attr['use_locking']
		elif node.op == 'AssignAdd':
			node.op = 'Add'
			if 'use_locking' in node.attr: 
				del node.attr['use_locking']


def get_latest_block_node(block_name, graph_nodes):
	i = 0
	latest_node = None
	while i < len(graph_nodes):
		if not graph_nodes[i].op == 'Const':
			if graph_nodes[i].name.startswith(block_name + '/') or graph_nodes[i].name == block_name:
				break
		i += 1
	while i < len(graph_nodes):
		if not graph_nodes[i].op == 'Const':
			if graph_nodes[i].name.startswith(block_name + '/') or graph_nodes[i].name == block_name:
				latest_node = graph_nodes[i]
		i += 1
	return latest_node

def get_first_block_node(block_name, graph_nodes):
	for n in graph_nodes:
		if not n.op == 'Const':
			if n.name.startswith(block_name + '/') or n.name == block_name:
				return n
	return None


def modify_graph_node_names(graph_def, node_name_map):
	modified_list = set()
	for n in graph_def.node:
		if n.name in node_name_map:
			n.name = node_name_map[n.name]
			modified_list.add(n.name)
		for i in range(len(n.input)):
			if n.input[i] in node_name_map:
				n.input[i] = node_name_map[n.input[i]]
				modified_list.add(n.name)
	return modified_list


def freeze_block(model_dir, block_name_map, output_path=None):
	"""Extract the sub graph defined by the output nodes and convert 
	all its variables into constant 
	Args:
		model_dir: the root folder containing the checkpoint state file
		output_block_names: a list, containing all the output node's names
	"""
	if not tf.gfile.Exists(model_dir):
		logTracker.logException('Export directory does not exists: ' + model_dir)

	# We retrieve our checkpoint fullpath
	checkpoint = tf.train.get_checkpoint_state(model_dir)
	input_checkpoint = checkpoint.model_checkpoint_path
	
	# We clear devices to allow TensorFlow to control on which device it will load operations
	clear_devices = True

	# We start a session using a temporary fresh Graph
	with tf.Session(graph=tf.Graph()) as sess:
		# We import the meta graph in the current default Graph
		saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

		# We restore the weights
		saver.restore(sess, input_checkpoint)
		
		graph_def = sess.graph.as_graph_def()
		fix_batch_norm_problem(graph_def)

		graph_nodes = list(graph_def.node)
		node_name_map = {}
		
		for block_name in block_name_map['input']:
			first_node = get_first_block_node(block_name, graph_nodes)
			if first_node is None:
				logTracker.logException('Not found first node from block: ' + str(block_name))
			
			new_name = block_name_map['input'][block_name]
			node_name_map[first_node.name] = new_name

		output_node_names = []
		for block_name in block_name_map['output']:
			latest_node = get_latest_block_node(block_name, graph_nodes)
			if latest_node is None:
				logTracker.logException('Not found latest node from block: ' + str(block_name))

			new_name = block_name_map['output'][block_name]
			if latest_node.name in node_name_map:
				logTracker.logException('Duplicated node name: ' + str(latest_node.name))
			node_name_map[latest_node.name] = new_name
			output_node_names.append(new_name)

		modify_graph_node_names(graph_def, node_name_map)
		logTracker.log('\nNode mapping')
		logTracker.log('-' * 30)
		for node_name in node_name_map:
			logTracker.log('{}  :  {}'.format(node_name, node_name_map[node_name]))
		logTracker.log('')

		node_names = [n.name for n in graph_def.node]
		if len(node_names) != len(set(node_names)):
			logTracker.logException('Duplicated node name')

		# We use a built-in TF helper to export variables to constants
		output_graph_def = tf.graph_util.convert_variables_to_constants(
			sess, # The session is used to retrieve the weights
			graph_def, # The graph_def is used to retrieve the nodes 
			output_node_names # The output node names are used to select the usefull nodes
		) 
		# We precise the file fullname of our freezed graph
		if output_path is None:
			output_path = os.path.join(os.path.dirname(input_checkpoint), "frozen_model.pb")

		# Finally we serialize and dump the output graph to the filesystem
		with tf.gfile.GFile(output_path, 'wb') as f:
			f.write(output_graph_def.SerializeToString())
		logTracker.log('{} operations in the final graph.'.format(len(output_graph_def.node)))
		save_json(block_name_map, os.path.join(getParentPath(output_path), 'block_name_map.json'))

	return output_graph_def, output_path


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", help="Model checkpoint directory path")
	parser.add_argument("--outmap", help="Block name mapping JSON file path")
	args = parser.parse_args()
	model_dir = args.path
	block_name_map = load_json(args.outmap)
	freeze_block(model_dir, block_name_map)
