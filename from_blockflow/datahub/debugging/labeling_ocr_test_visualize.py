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
from visualizing.visualize import visualize_sample, get_class_color_map


if __name__ == '__main__':
	project_dir = os.path.join('data', 'invoice')
	data_dir = os.path.join(project_dir, 'generated', '20190830-020153')
	img_dir = os.path.join(project_dir, 'stuff', 'imgs')
	f_names = [
	# From inference debug (valset also)
		'19_0.json',
		'1_ヤマト運輸株式会社_0.json',
		'24_0.json',
		'34 (2)_0.json',
	# From trainset
		'0785_109_26.json',
	# From valset
		'31 (2)_0.json',
		'32_0.json',
		'19_0.json',
		'1 (2)_0.json',
		'1609LeBAC請求書_0.json',
		'1612サムライ請求書_0.json',
		'20180704134140-0001_0.json',
		'0790_060_15.json',
		'0792_001_37.json',
	]

	#f_names = ['19_0.json']

	qa_dir = os.path.join(data_dir, 'qa-labels-fk-mapped')
	label_json_paths = [os.path.join(qa_dir, f_name) for f_name in f_names]
	
	ocr_dir = os.path.join(data_dir, 'ocr-output-standard')
	
	class_path = os.path.join(project_dir, 'classes.json')
	classes = load_json(class_path)

	corpus_path = os.path.join(data_dir, 'corpus-info', 'corpus-old.json')
	#corpus_path = os.path.join(data_dir, 'corpus-info', 'corpus.json')
	#corpus_path = os.path.join(project_dir, 'generated','20190822-150651', '20190823-023131-corpus', 'corpus.json')
	corpus = load_json(corpus_path)

	max_width = 2500
	content_type = 'text'
	class_color_map = get_class_color_map(classes)

	for label_path in label_json_paths:
		f_name = getBasename(label_path)
		image_path = os.path.join(img_dir, f_name.replace('.json', '.png'))
		if not os.path.exists(image_path):
			image_path = os.path.join(img_dir, f_name.replace('.json', '.jpg'))

		ocr_path = os.path.join(ocr_dir, f_name)	
		if not os.path.exists(image_path) or not os.path.exists(ocr_path):
			continue
		
		qa_sample = load_json(label_path)
		ocr_sample = load_json(ocr_path)
		
		ocr_sample = generate_labeled_ocr_sample(ocr_sample, qa_sample, classes, corpus)

		image = visualize_sample(qa_sample, ocr_sample, image_path, max_width, class_color_map, content_type, corpus)
		image.show()
		
		input('')


