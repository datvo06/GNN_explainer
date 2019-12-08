# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil
import time

from math import sin, cos, radians
from collections import defaultdict
from frozendict import frozendict
from copy import deepcopy
from random import shuffle

from chi_lib.library import *
from processing.label_ocr_samples import generate_labeled_ocr_sample
from visualizing.utils import get_samples_max_size
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar


def adjust_location(textline, new_content, extend_percentage):
	#textline_w, textline_h = get_region_size(textline)
	xt, yt, xb, yb = get_topleft_bottomright(textline)
	textline_w = xb - xt
	textline_h = yb - yt
	content_w, content_h = new_content['size']
	#form_w, form_h = new_content['sample_size']
	scale = textline_h / content_h
	content_h = content_h * scale
	content_w = int(np.ceil(min(textline_w * (1 + extend_percentage), content_w  * scale)))
	#xt, yt, _, _ = get_topleft_bottomright(textline)
	xb, yb = int(xt + content_w), int(yt + content_h)
	return [[xt, yt], [xb, yt], [xb, yb], [xt, yb]]


def get_new_content(textline, database_map, extend_percentage):
	max_length = int(np.ceil(len(textline['text']) * (1 + extend_percentage)))
	min_length = int(np.ceil(len(textline['text']) * (1 - extend_percentage)))
	def is_valid(content):
		text = content['text']
		return len(text) <= max_length and len(text) >= min_length

	label_info = textline['label_info']
	fk_type = label_info['key_type']
	fk = label_info['formal_key']
	contents = database_map[fk_type][fk]
	# This trick is to speed-up the picking process. Briefly, instead of checking the validity of all content in the database, we randomly pick one by one content (m times) and check its validity, if it passes, return the content. 
	# You can wonder why not just setting m = n and do the random picking n times? It is not good because we need to re-check the index of of the already checked content, and it becomes larger by time.
	trials = 200
	n = len(contents)
	checked = set()
	while trials > 0:
		trials -= 1
		idx = np.random.randint(0, n)
		while idx in checked:
			idx = np.random.randint(0, n)
		res = contents[idx]
		if is_valid(res):
			return res
		checked.add(idx)
	# If the number of trials exceeds m times, it means the valid contents is quite rate, thus it better to scan through the whole database to get the valid data.
	contents = [content for content in contents if is_valid(content)]
	return contents[np.random.randint(0, len(contents))]


def generate_new_textline_content(textline, database_map, extend_percentage):
	new_content = get_new_content(textline, database_map, extend_percentage)
	label_info = textline['label_info']
	return {
		'location' 	 : adjust_location(textline, new_content, extend_percentage),
		'type'		 : 'textline',
		'text'		 : new_content['text'],
		'label_info' : {
			'formal_key' : label_info['formal_key'],
			'key_type'	 : label_info['key_type'],
		}
	}


def shuffle_textlines_content(sample, database_map, extend_percentage):
	res = []
	for textline in sample:
		if len(textline['text'].strip('\n\r\t')) > 0:
			textline = generate_new_textline_content(textline, database_map, extend_percentage)
		else:
			textline = deepcopy(textline)
		res.append(textline)
	return res















