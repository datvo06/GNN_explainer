# encoding: utf-8


import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil
import cv2

from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from frozendict import frozendict
from fontTools.ttLib import TTFont

from chi_lib.library import *
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar
from processing.label_ocr_samples import get_topleft_bottomright


font_path = os.path.join('visualizing', 'font', "SimSun.ttf")
textline_font = ImageFont.truetype(font_path, 13)
def visualize_multi_samples(samples, max_width):
	w, h = get_samples_max_size(samples)
	w, h = w+30, h+30
	image = np.zeros((h, w, 3), dtype="uint8")
	image = Image.fromarray(image)

	w, h = image.size
	if max_width is None:
		max_width = w * 2
	scale = max_width / (w * 2)
	w = int(w * scale)
	h = int(h * scale)
	image = image.resize((w, h), Image.ANTIALIAS)

	plots = []
	for sample in samples:
		temp = np.array(image)
		plots.append(plot_boxes(sample, image=temp, scale=scale))
	
	image = np.concatenate(plots, axis=1).astype(np.uint8)
	image = Image.fromarray(image)
	return image


def plot_boxes(sample, image, scale=1.0):
	h, w, _ = image.shape
	for textline in sample:
		pts = [(int(p[0]*scale), int(p[1]*scale)) for p in textline['location']]
		pts = np.array(pts, dtype=np.int32)
		#x1, y1, x2, y2 = get_topleft_bottomright(textline)
		#p1 = (int(x1*scale), int(y1*scale))
		#p2 = (int(x2*scale), int(y2*scale))
		color = (0, 255, 0)
		cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)

	image = Image.fromarray(image)
	draw = ImageDraw.Draw(image)	
	for textline in sample:
		x1, y1, x2, y2 = get_topleft_bottomright(textline)
		p1 = (int(x1*scale), int(y1*scale))
		draw.text(p1, textline['text'], font=textline_font, fill=(0, 255, 0))	
	image = np.array(image)
	cv2.rectangle(image, (0, 0), (w, h), (0, 0, 150), 2, cv2.LINE_AA)
	return image


def get_samples_max_size(samples):
	max_w = 0
	max_h = 0
	for sample in samples:
		if not sample is None:
			pts = []
			for tl in sample:
				pts.extend(tl['location'])
			w = max([p[0] for p in pts])
			h = max([p[1] for p in pts])
			max_w = max(max_w, w)
			max_h = max(max_h, h)
	return max_w, max_h


def add_plot_title(plot_image, title):
	import pdb
	# pdb.set_trace()
	p_h, p_w, _ = plot_image.shape
	title_image = np.full((15, p_w, 3), 150)
	t_h, t_w, _ = title_image.shape
	cv2.rectangle(cv2.UMat(title_image).get(), (0, 0), (int(t_w), (t_h)), (0, 0, 150), 2, 5)
	cv2.putText(cv2.UMat(title_image).get(), title, (int(t_w/2+2), int(t_h/2+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), thickness=1, lineType=cv2.LINE_8)
	return np.concatenate((title_image, plot_image), axis=0)


def plot_class_colors(class_color_map, width, height):
	label_blocks = []
	for class_name in sorted(class_color_map):
		image = np.zeros((h, w, 3), dtype="uint8")
		image = Image.fromarray(image)


def get_class_color_map(classes):
	def get_rgb(value):
		value = int(value)
		b =  value % 256
		g = value // 256 % 256
		r =   value // 256 // 256 % 256
		return r, g, b

	classes = list(classes)
	res = {}
	max_int = 9999
	i = 423
	max_trials = 20
	trials = max_trials
	while len(classes) > 0 and trials > 0:
		i = ((i + 34) % max_int) * ((i + 435) % max_int) * ((int(i/2) + 73) % max_int)
		color = get_rgb(i)
		temp = np.array(color, dtype=np.float)
		avg = np.mean(temp, dtype=np.float)
		max_diff = np.max(np.abs(temp - avg))
		min_val = np.min(temp)
		if max_diff < 10 and min_val < 20:
			trials -= 1
			continue
		trials = max_trials
		res[classes[0]] = color
		classes.pop(0)
	return res
