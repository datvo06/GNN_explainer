import os, sys, json
sys.path.append('.') 

import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil
import re
import unicodedata

from chi_lib.library import *
from collections import defaultdict
from frozendict import frozendict
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar


# Normalizing reference: https://www.fileformat.info/info/unicode/category/index.htm


def get_unicode_bracket_pairs():
	brackets_list = ''
	other_brackets_list = ''
	for c in map(chr, range(sys.maxunicode + 1)):
		# Ref: https://stackoverflow.com/questions/13535172/list-of-all-unicodes-open-close-brackets
		if unicodedata.category(c) in ['Ps', 'Pe', 'Pi', 'Pf']:
			if unicodedata.mirrored(c):
				brackets_list += c
			else:
				other_brackets_list += c

	if len(brackets_list) % 2 != 0 or len(other_brackets_list) % 2 != 0:
		logTracker.log('Non-symmetric bracket list:')
		logTracker.log(' - Bracket : ' + brackets_list)
		logTracker.log(' - Other   : ' + other_brackets_list)

	brackets_list += other_brackets_list
	lefts = []
	rights = []
	for i in range(0, len(brackets_list), 2):
		c1 = brackets_list[i]
		c2 = brackets_list[i+1]
		lefts.append(c1)
		rights.append(c2)
	return lefts, rights


def get_unicode_chars_by_categories(categories):
	res = ''
	other = ''
	for c in map(chr, range(sys.maxunicode + 1)):
		if unicodedata.category(c) in categories:
			if unicodedata.mirrored(c):
				res += c
			else:
				other += c
	return res + other


def get_unicode_chars_by_similar_names(name_parts, categories=None):
	def is_ok(name):
		if name is None:
			return False
		if name in name_parts:
			return True
		for name_part in name_parts:
			if name_part in name:
				return True
		return False

	res = ''
	other = ''

	if categories:
		cur_str = get_unicode_chars_by_categories(categories)
	else:
		cur_str = map(chr, range(sys.maxunicode + 1)) 
	for c in cur_str:
		try:
			name = unicodedata.name(c)
		except:
			name = None
		if is_ok(name):
			if unicodedata.mirrored(c):
				res += c
			else:
				other += c

	return res + other


def generate_pattern_from_list(strs):
	return r'[' + re.escape(r''.join(strs)) + r']'


def generate_normalizers():
	left_brackets, right_brackets = get_unicode_bracket_pairs()
	dashes = get_unicode_chars_by_categories(['Pd'])
	spaces = get_unicode_chars_by_categories(['Zs', 'Zl', 'Zp'])
	dots = get_unicode_chars_by_similar_names(['DOT ', ' DOT', ' STOP', 'STOP '], ['Po'])
	#modifiers = get_unicode_chars_by_categories(['Sk'])
	return [
		('0', re.compile(r'[0-9]')),
		('"', re.compile(r'\'')),
		(',', re.compile(r'\;')),
		('-', re.compile(r'_')),
		(' ', re.compile(r'[\t\n\r]')),
		('-', re.compile(generate_pattern_from_list(dashes))),
		(' ', re.compile(generate_pattern_from_list(spaces))),
		('.', re.compile(generate_pattern_from_list(dots))),
		('(', re.compile(generate_pattern_from_list(left_brackets))),
		(')', re.compile(generate_pattern_from_list(right_brackets))),
	]


normalizers = generate_normalizers()
def normalize_text(text, corpus=None):
	text = text.lower()
	text = unicodedata.normalize('NFKC', text)
	for target_unichr, normalizer in normalizers:
		text = normalizer.sub(target_unichr, text)
	if not corpus is None:
		text = ''.join([t if t in corpus else '�' for t in text])
	return text


	