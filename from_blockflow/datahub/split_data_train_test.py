import os, sys, json
import glob                                                           
import cv2 
import shutil
import argparse
import numpy as np
from chi_lib.library import *
from collections import defaultdict
from chi_lib.ProgressBar import ProgressBar


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='Labels directory path', required=True)
	parser.add_argument('--train', nargs='+', help='Train lst file paths', required=True)
	parser.add_argument('--val', nargs='+', help='Test lst file paths', required=True)
	args = parser.parse_args()

	labelsDir = args.path
	trainLstPaths = args.train
	testLstPaths = args.val
	
	data = defaultdict(lambda : set())
	for lstPath in trainLstPaths:
		data['train'].update(load_path_list(lstPath))

	for lstPath in testLstPaths:
		data['test'].update(load_path_list(lstPath))

	res_path = os.path.join(getParentPath(labelsDir), getTodayDatetimeString() + '_split_' + getBasename(labelsDir))
	createDirectory(res_path)
	for lst in data:
		logTracker.log(lst)
		resDir = os.path.join(res_path, lst + '_data')
		createDirectory(resDir)
		progress = ProgressBar(name='Copying', maxValue=len(data[lst]))
		for fName in data[lst]:
			fPath = os.path.join(labelsDir, fName)
			#shutil.copytree(os.path.join(labelsDir, fName), resDir)
			shutil.copy(os.path.join(labelsDir, fName), resDir)
			progress.increase()
		progress.done()