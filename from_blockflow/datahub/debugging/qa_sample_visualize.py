# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil

from chi_lib.library import *
from collections import defaultdict
from frozendict import frozendict
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar
from copy import deepcopy

from processing.label_ocr_samples import generate_labeled_ocr_sample
from visualizing.visualize import visualize_samples, get_class_color_map


if __name__ == '__main__':
	project_dir = os.path.join('data', 'invoice')
	data_dir = os.path.join(project_dir, 'generated', '20190924-152618')
	img_dir = os.path.join(project_dir, 'stuff', 'imgs')
	# f_names = [
	# # From inference debug (valset also)
	# 	'19_0.json',
	# 	'1_ヤマト運輸株式会社_0.json',
	# 	'24_0.json',
	# 	'34 (2)_0.json',
	# # From trainset
	# 	'0785_109_26.json',
	# # From valset
	# 	'31 (2)_0.json',
	# 	'32_0.json',
	# 	'19_0.json',
	# 	'1 (2)_0.json',
	# 	'1609LeBAC請求書_0.json',
	# 	'1612サムライ請求書_0.json',
	# 	'20180704134140-0001_0.json',
	# 	'0790_060_15.json',
	# 	'0792_001_37.json',
	# ]

	#f_names = ['19_0.json']

	qa_dir = os.path.join(data_dir, 'qa-labels-fk-mapped')
	label_json_paths = loadValidFiles(qa_dir, 'json', keepFilePath=True)

	#label_json_paths = sorted(label_json_paths)
	
	class_path = os.path.join(project_dir, 'classes.json')
	classes = load_json(class_path)

	corpus_path = os.path.join(data_dir, 'corpus-info', 'corpus.json')
	#corpus_path = os.path.join(data_dir, 'corpus-info', 'corpus.json')
	#corpus_path = os.path.join(project_dir, 'generated','20190822-150651', '20190823-023131-corpus', 'corpus.json')
	corpus = load_json(corpus_path)

	max_width = 1800
	content_type = 'text'
	class_color_map = get_class_color_map(classes)

	for label_path in label_json_paths:
		f_name = getBasename(label_path)
		image_path = os.path.join(img_dir, f_name.replace('.json', '.png'))
		if not os.path.exists(image_path):
			image_path = os.path.join(img_dir, f_name.replace('.json', '.jpg'))

		print(image_path)
		qa_sample = load_json(label_path)
		image = visualize_samples([qa_sample], ['QA'], image_path, max_width, class_color_map, content_type, corpus)
		image.show()
		
		input('')


