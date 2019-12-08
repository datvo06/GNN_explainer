import os, sys, json
sys.path.append('.') 

import numpy as np
import argparse
import random

from collections import defaultdict
from frozendict import frozendict
from copy import deepcopy

from chi_lib.library import *
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar


class FileFilter:
	def __init__(self, f_paths, keepFilePath=False):
		self.f_paths = tuple(f_paths)
		self.keepFilePath = keepFilePath
		if self.keepFilePath is False:
			self.f_paths = tuple([getBasename(f_path) for f_path in self.f_paths])

	def mask(self, target_f_paths):
		target_f_paths = target_f_paths
		if self.keepFilePath is False:
			target_f_paths = [getBasename(f_path) for f_path in target_f_paths]

		target_f_paths = set(target_f_paths)
		logTracker.log('Getting mask for ' + str(len(target_f_paths)) + ' files in ' + str(len(self.f_paths)))
		progress = ProgressBar('Getting', len(self.f_paths))
		indices = []
		for i, f_path in enumerate(self.f_paths):
			if f_path in target_f_paths:
				indices.append(i)
			progress.increase()
		progress.done()
		logTracker.log('Got ' + str(len(indices)) + ' / ' + str(len(target_f_paths)) + ' / ' + str(len(self.f_paths)) + ' files')
		return get_masks(indices, len(self.f_paths))


	def mask_lst(self, lst_path):
		logTracker.log('Getting mask from ' + lst_path)
		target_f_paths = load_path_list(lst_path)
		return self.mask(target_f_paths)


def get_masks(sample_indices, max_n):
	return [True if i in sample_indices else False for i in range(max_n)]