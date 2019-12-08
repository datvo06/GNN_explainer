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

from chi_lib.library import *
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar



def rotate_polygon(pts, degrees):
	""" Rotate polygon the given angle about its center. """
	theta = radians(degrees)  # Convert angle to radians
	cosang, sinang = cos(theta), sin(theta)

	xs = [p[0] for p in pts]
	ys = [p[1] for p in pts]

	cx = (max(xs) + min(xs)) / 2
	cy = (max(ys) + min(ys)) / 2

	new_pts = []
	for p in pts:
		x, y = p[0], p[1]
		tx, ty = x-cx, y-cy
		new_x = ( tx*cosang + ty*sinang) + cx
		new_y = (-tx*sinang + ty*cosang) + cy
		new_pts.append((new_x, new_y))
	min_x = min([p[0] for p in new_pts])
	min_y = min([p[1] for p in new_pts])
	min_x = np.abs(min(min_x, 0))
	min_y = np.abs(min(min_y, 0))
	new_pts = [(int(p[0]+min_x), int(p[1]+min_y)) for p in new_pts]
	return new_pts


def rotate_sample(sample, degrees):
	sample = deepcopy(sample)
	sample_location_map = {}
	pts = []
	for i, textLine in enumerate(sample):
		location = textLine['location']
		# The starting position and the length of points list for each sample in the overall points list
		sample_location_map[i] = (len(pts), len(location))
		pts.extend(location)
		
	new_pts = rotate_polygon(pts, degrees)
	if len(pts) != len(new_pts):
		logTracker.logException('Inconsistent return: ' + str(len(pts)) + ' - ' + str(len(new_pts)))
	
	for i, textLine in enumerate(sample):
		s, l = sample_location_map[i]
		textLine['location'] = new_pts[s:s+l]
	return sample










