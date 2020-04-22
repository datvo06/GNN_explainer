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
from normalizing.normalize_text import normalize_text
from visualizing.utils import *

font_path = os.path.join('visualizing', 'font', "SimSun.ttf")
textline_font = ImageFont.truetype(font_path, 13)

textline_ttf_font = TTFont(font_path)   # specify the path to the font in question
def char_in_font(unicode_char):
	for cmap in textline_ttf_font['cmap'].tables:
		if cmap.isUnicode():
			if ord(unicode_char) in cmap.cmap:
				return True
	return False


def plot_textline_boxes(sample, image, scale=1.0, class_color_map={}, content_type='key', corpus=None):
	content_type = content_type.strip('\n\r\t')
	w, h = image.size
	image = np.array(image)
	if not sample is None:
		alpha = 0.5
		overlay = image.copy()
		try:
			for textline in sample:
				x1, y1, x2, y2 = get_topleft_bottomright(textline)
				p1 = (int(x1*scale), int(y1*scale))
				p2 = (int(x2*scale), int(y2*scale))
				fk = textline['label_info']['formal_key']
				color = (0, 0, 0)
				if fk in class_color_map:
					color = class_color_map[fk]
					if textline['label_info']['key_type'] == 'value':
						# -1 is to fill the color on the rectangle
						cv2.rectangle(overlay, p1, p2, color, -1, cv2.LINE_AA)

					# We also want avoid changing the border line color (due to the below addWeighted step)
					cv2.rectangle(overlay, p1, p2, color, 2, cv2.LINE_AA)

				# This step is to remove the transparency of Non required classes' rectangles
				# 2 is the thickness of line
				cv2.rectangle(image, p1, p2, color, 2, cv2.LINE_AA)
				
				#draw.rectangle((p1, p2), fill=0, outline=line_color)
				#cv2.putText(image, text, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 0, 0), thickness=1, lineType=cv2.LINE_8)

			cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
			image = Image.fromarray(image)
			draw = ImageDraw.Draw(image)
				
			for textline in sample:	
				x1, y1, x2, y2 = get_topleft_bottomright(textline)
				p1 = (int(x1*scale), int(y1*scale))
				if content_type == 'key':
					label_info = textline['label_info']
					content = label_info['key_type'] + '#' + label_info['formal_key']
				elif content_type == 'text':
					content = textline['text']
					if not corpus is None:
						content = normalize_text(content) 
						content = [c if char_in_font(c) else '?' for c in content]
						content = [c if c in corpus else '*' for c in content]
						content = ''.join(content)
				
				draw.text(p1, content, font=textline_font, fill=(255, 0, 0))	
			image = np.array(image)
		except:
			for textline in sample:
				x1, y1, x2, y2 = get_topleft_bottomright(textline)
				p1 = (int(x1*scale), int(y1*scale))
				p2 = (int(x2*scale), int(y2*scale))
				fk = textline['key_type']
				color = (0, 0, 0)
				if fk in class_color_map:
					color = class_color_map[fk]
					if textline['key_type'] == 'value':
						# -1 is to fill the color on the rectangle
						cv2.rectangle(overlay, p1, p2, color, -1, cv2.LINE_AA)

					# We also want avoid changing the border line color (due to the below addWeighted step)
					cv2.rectangle(overlay, p1, p2, color, 2, cv2.LINE_AA)

				# This step is to remove the transparency of Non required classes' rectangles
				# 2 is the thickness of line
				cv2.rectangle(image, p1, p2, color, 2, cv2.LINE_AA)
				
				#draw.rectangle((p1, p2), fill=0, outline=line_color)
				#cv2.putText(image, text, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 0, 0), thickness=1, lineType=cv2.LINE_8)

			cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
			image = Image.fromarray(image)
			draw = ImageDraw.Draw(image)
				
			for textline in sample:	
				x1, y1, x2, y2 = get_topleft_bottomright(textline)
				p1 = (int(x1*scale), int(y1*scale))
				if content_type == 'key':
					content = textline['key_type'] + '#' + textline['type']
				elif content_type == 'text':
					content = textline['text']
					if not corpus is None:
						content = normalize_text(content) 
						content = [c if char_in_font(c) else '?' for c in content]
						content = [c if c in corpus else '*' for c in content]
						content = ''.join(content)
				
				draw.text(p1, content, font=textline_font, fill=(255, 0, 0))	
			image = np.array(image)

	cv2.rectangle(image, (0, 0), (w, h), (0, 0, 150), 2, cv2.LINE_AA)
	return image


def visualize_sample(qa_sample, ocr_sample, image_path, max_width, class_color_map, content_type, corpus=None):
	if image_path is None:
		w, h = get_samples_max_size([qa_sample, ocr_sample])
		w, h = w+30, h+30
		image = np.zeros((h, w, 3), dtype="uint8")
		image = Image.fromarray(image)
	else:
		image = Image.open(image_path)
	
	w, h = image.size
	if max_width is None:
		max_width = w * 2
	scale = max_width / (w * 2)
	w = int(w * scale)
	h = int(h * scale)
	image = image.resize((w, h), Image.ANTIALIAS)
	label_plot = plot_textline_boxes(qa_sample, image=image, scale=scale, class_color_map=class_color_map, content_type='key', corpus=corpus)
	ocr_plot = plot_textline_boxes(ocr_sample, image=image, scale=scale, class_color_map=class_color_map, content_type='key', corpus=corpus)
	label_plot = add_plot_title(label_plot, 'QA')
	ocr_plot = add_plot_title(ocr_plot, 'OCR')
	#class_color_plot = plot_class_colors(class_color_map, width=50, height=ocr_plot.shape[0])
	image = np.concatenate((ocr_plot, label_plot), axis=1).astype(np.uint8)
	image = Image.fromarray(image)
	return image


def visualize_samples(samples, titles, image_path, max_width, class_color_map, content_type, corpus=None):
	if len(samples) != len(titles):
		logTracker.logException('Samples size must equal to titles size: {} - {}'.format(len(samples), len(titles)))

	if image_path is None:
		w, h = get_samples_max_size(samples)
		w, h = w+30, h+30
		image = np.zeros((h, w, 3), dtype="uint8")
		image = Image.fromarray(image)
	else:
		image = Image.open(image_path)
	
	w, h = image.size
	if max_width is None:
		max_width = w * 2
	scale = max_width / (w * 2)
	w = int(w * scale)
	h = int(h * scale)
	image = image.resize((w, h), Image.ANTIALIAS)
	plots = []
	for i, sample in enumerate(samples):
		cur_plot = plot_textline_boxes(sample, image=image, scale=scale, class_color_map=class_color_map, content_type=content_type, corpus=corpus)
		plots.append(add_plot_title(cur_plot, str(titles[i])))
	#class_color_plot = plot_class_colors(class_color_map, width=50, height=ocr_plot.shape[0])
	image = np.concatenate(plots, axis=1).astype(np.uint8)
	image = Image.fromarray(image)
	return image	


def visualize_samples_from_files(label_json_paths, ocr_json_paths, image_paths, max_width, classes, content_type, corpus, res_dir):
	createDirectory(res_dir)
	class_color_map = {}
	if classes:
		class_color_map = get_class_color_map(classes)
	label_map = get_json_file_map_from_file_paths(label_json_paths)
	ocr_map = get_json_file_map_from_file_paths(ocr_json_paths)
	image_path_map = {}
	for f_path in image_paths:
		f_name = getBasename(f_path)
		f_name = convert_img_to_json_name(f_name)
		image_path_map[f_name] = f_path

	for f_name in ocr_map.keys():
		new_name = f_name.replace('_lc.json', '.json')
		ocr_map[new_name] = ocr_map.pop(f_name)
	
	f_names = set(ocr_map.keys()).union(set(label_map.keys()))
	pair_map = {}
	for f_name in f_names:
		pair_map[f_name] = (
			label_map[f_name] if f_name in label_map else None,
			ocr_map[f_name] if f_name in ocr_map else None,
			image_path_map[f_name] if f_name in image_path_map else None
		)
	logTracker.log('Saving ' + str(len(pair_map)) + ' visualizing images to ' + res_dir)
	progress = ProgressBar('Saving', len(pair_map))
	for f_name, pair in pair_map.items():
		progress.increase()
		if pair[2] is None or pair[1] is None or pair[0] is None:
			continue
		image = visualize_sample(pair[0], pair[1], pair[2], max_width=max_width, class_color_map=class_color_map, content_type=content_type, corpus=corpus)
		#image.show()
		# pdb.set_trace()
		image.save(os.path.join(res_dir, f_name.replace('.json', '.jpg')))
	progress.done()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--qa', help='Standard QA samples directory path', required=True)
	parser.add_argument('--ocr', help='Standard OCR-Linecut samples directory path', default=None)
	parser.add_argument('--corpus', help='Standard OCR-Linecut samples directory path', default=None)
	parser.add_argument('--img', help='Form images directory path', default=None)
	parser.add_argument('--classes', help='Classes file path', default=None)
	parser.add_argument('--type', help='Content type [key / text]', default='text')
	parser.add_argument('--selected', nargs='+', help='Selected file names .lst paths', default=[])
	parser.add_argument('--res', help='Result directory path', required=True)

	args = parser.parse_args()

	label_dir_path = args.qa
	ocr_dir_path = args.ocr
	image_dir_path = args.img
	
	classes = None
	if args.classes:
		classes = load_json(args.classes)

	if args.corpus:
		corpus = load_json(args.corpus)

	label_json_paths = loadValidFiles(label_dir_path, 'json', keepFilePath=True)
	label_json_paths = filter_file_paths_from_path_lists(label_json_paths, args.selected)
		
	ocr_json_paths = []
	if ocr_dir_path:
		ocr_json_paths = loadValidFiles(ocr_dir_path, 'json', keepFilePath=True)
		ocr_json_paths = filter_file_paths_from_path_lists(ocr_json_paths, args.selected)

	image_paths = []
	if image_dir_path:
		image_paths = loadValidFiles(image_dir_path, 'jpg', keepFilePath=True)
		image_paths += loadValidFiles(image_dir_path, 'png', keepFilePath=True)
		#image_paths = filter_file_paths_from_path_lists(image_paths, args.selected)		
	
	content_type = args.type

	visualize_samples_from_files(label_json_paths, ocr_json_paths, image_paths, 2000, classes, content_type, corpus, args.res)
		



