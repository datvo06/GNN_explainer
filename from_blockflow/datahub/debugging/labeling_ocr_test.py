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

from processing.label_ocr_samples import label_ocr_samples


if __name__ == '__main__':
	project_dir = os.path.join('data', 'okaya')
	data_dir = os.path.join(project_dir, 'generated', '20190823-100609')

	qa_dir = os.path.join(data_dir, 'qa-labels-fk-mapped')
	label_json_paths = [os.path.join(qa_dir, '【納品書】ナカジマ鋼管①_1.json')]
	
	ocr_dir = os.path.join(data_dir, 'ocr-output-standard')
	ocr_json_paths = [os.path.join(ocr_dir, '【納品書】ナカジマ鋼管①_1.json')]

	classes = None
	class_path = os.path.join(project_dir, 'classes.json')
	if class_path:
		with open(class_path, 'r', encoding='utf8') as f:
			classes = json.loads(f.read())

	res_dir_path = os.path.join('debugging', 'labeling_ocr_test_res')
	
	label_ocr_samples(label_json_paths, ocr_json_paths, res_dir_path, classes)
		



