# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil

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


def get_merged_location(textlines):
	pts = []
	for textline in textlines:
		pts.extend(textline['location'])
	xs = [p[0] for p in pts]		
	ys = [p[1] for p in pts]
	x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
	return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def get_merged_text(textlines):
	res = ''
	for textline in textlines:
		res += textline['text']
	return res
	

def merge_textlines(textlines):
	return {
		'location'  : get_merged_location(textlines),
		'type'		: 'textline',
		'text'		: get_merged_text(textlines)
	}


def merge_textlines_label(textlines, classes, corpus, merge_rate=None):
	merged_textline = merge_textlines(textlines)
	if not merge_rate is None and merge_rate > 0:
		np.random.shuffle(textlines)
		length = int(len(textlines) * merge_rate)
		if length > 2:
			textlines = textlines[:length]
	res = generate_labeled_ocr_sample([merged_textline], textlines, classes, corpus)
	return res[0]


def random_merge_textlines(sample, classes, corpus, max_dist_pct_w, merge_rate):
	sample = deepcopy(sample)
	sample = [tl for tl in sample if len(tl['text'].strip('\n\r\t')) > 0]
	w, h = get_samples_max_size([sample])
	center_map = {id(tl) : get_center(tl) for tl in sample}

	#sample = [tl for tl in sample if center_map[id(tl)][1] > 1700 and center_map[id(tl)][1] < 2100]
	# Spliting textlines into groups, each group contains a set of textlines lying on the same line
	sample = sorted(sample, key=lambda x: center_map[id(x)][1])
	lines = []
	while len(sample) > 0:
		line = [sample.pop(-1)]
		for i in range(len(sample)-1, -1, -1):
			textline = sample[i]
			# Compare with the all y-position 
			max_angle = 0.0
			for j in range(len(line)):		
				center_1 = center_map[id(line[j])]
				center_2 = center_map[id(textline)]
				center_dist = dist(center_1[0], center_1[1], center_2[0], center_2[1])
				w_dist = np.abs(center_1[0] - center_2[0])
				angle = 0.0
				if center_dist > 1e-4:
					angle = np.degrees(np.arccos(float(w_dist) / float(center_dist)))
				max_angle = max(angle, max_angle)
			if max_angle < 3:
				line.append(sample.pop(i))
		lines.append(line)

	# Merging the textlines which are close together, and on the same line
	new_sample = []
	for line in lines:
		line = sorted(line, key=lambda x: center_map[id(x)][0])
		merged = [line[0]]
		for textline in line[1:]:
			# Compare with the highest x-position in the line
			tl_dist = textlines_distance(merged[-1], textline)
			tl_dist = float(tl_dist)/float(w)
			if tl_dist < max_dist_pct_w:
				merged.append(textline)
			else:
				is_merge = np.random.uniform(0, 1)
				if is_merge <= merge_rate:
					new_sample.append(merge_textlines_label(merged, classes, corpus, merge_rate))
				else:
					new_sample.extend(merged)
				merged = [textline]

		if len(merged) > 0:
			is_merge = np.random.uniform(0, 1)
			if is_merge <= merge_rate:
				new_sample.append(merge_textlines_label(merged, classes, corpus, merge_rate))
			else:
				new_sample.extend(merged)
			merged = None

	return new_sample

















