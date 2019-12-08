# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil
import re
import time

from copy import deepcopy
from chi_lib.library import *
from collections import defaultdict
from frozendict import frozendict
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar
from visualizing.stats_corpus import *

if __name__ == '__main__':
	project_dir = os.path.join('data', 'invoice')
	data_dir_path = os.path.join(project_dir, 'generated', '20190830-020153', 'qa-labels-fk-mapped')
	selected_lst_path = os.path.join(project_dir, 'train.lst')
	required_keys = load_json(os.path.join(project_dir, 'classes.json'))

	json_paths = loadValidFiles(data_dir_path, 'json', keepFilePath=True)
	json_paths = filter_file_paths_from_path_lists(json_paths, [selected_lst_path])
	samples = [load_json(f_path) for f_path in json_paths]

	key_type_corpus_map = get_corpus_map(samples, required_keys)
	intersection_map = get_corpus_intersection_map(key_type_corpus_map)
	print_intersection_count_map(intersection_map)


	