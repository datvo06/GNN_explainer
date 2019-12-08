import sys

import os 
import json
import itertools
import requests	
import csv
import numpy as np
import psutil
import shutil

from copy import deepcopy
from frozendict import frozendict
from scipy import stats
from collections import defaultdict
from chi_lib.LogTracker import LogTracker
from chi_lib.config import *
from datetime import datetime as dt

logTracker = LogTracker(LOG_DIR, IS_PRINT_LOG, IS_WRITE_LOG)

# def productDict(**kwargs):
# 	keys = kwargs.keys()
# 	vals = kwargs.values()
# 	for instance in itertools.product(*vals):
# 		yield dict(zip(keys, instance))


# Ref: https://stackoverflow.com/questions/3948873/prevent-function-overriding-in-python
class Access(type):
	__SENTINEL = object()
	def __new__(mcs, name, bases, class_dict):
		private = {key
				   for base in bases
				   for key, value in vars(base).items()
				   if callable(value) and mcs.__is_final(value)}
		if any(key in private for key in class_dict):
			raise RuntimeError('certain methods may not be overridden')
		return super().__new__(mcs, name, bases, class_dict)

	@classmethod
	def __is_final(mcs, method):
		try:
			return method.__final is mcs.__SENTINEL
		except AttributeError:
			return False

	@classmethod
	def final(mcs, method):
		method.__final = mcs.__SENTINEL
		return method


# Support functions


def get_closest_textline(target_textline, textlines):
	return min(textlines, key=lambda x: textlines_distance(target_textline, x))


def dist(x1, y1, x2, y2):
	return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def textlines_distance(textline1, textline2):
	x1, y1, x1b, y1b = get_topleft_bottomright(textline1)
	x2, y2, x2b, y2b = get_topleft_bottomright(textline2)

	left = x2b < x1
	right = x1b < x2
	bottom = y2b < y1
	top = y1b < y2

	if top and left:
		return dist(x1, y1b, x2b, y2)
	elif left and bottom:
		return dist(x1, y1, x2b, y2b)
	elif bottom and right:
		return dist(x1b, y1, x2, y2b)
	elif right and top:
		return dist(x1b, y1b, x2, y2)
	elif left:
		return x1 - x2b
	elif right:
		return x2 - x1b
	elif bottom:
		return y1 - y2b
	elif top:
		return y2 - y1b
	return 0.


def get_center(textline):
	pts = textline['location']
	xs = [p[0] for p in pts]
	ys = [p[1] for p in pts]
	return int(np.mean(xs)), int(np.mean(ys))


def get_topleft_bottomright(textline):
	points = textline['location']
	xs = [p[0] for p in points]		
	ys = [p[1] for p in points]
	return min(xs), min(ys), max(xs), max(ys)


def get_rectangle_area(loc):
	return (loc[2] - loc[0]) * (loc[3] - loc[1])


def get_intersection_area(loc1, loc2):
	x1 = max(loc1[0], loc2[0])
	y1 = max(loc1[1], loc2[1])
	x2 = min(loc1[2], loc2[2])
	y2 = min(loc1[3], loc2[3])
	return max(0, x2 - x1) * max(0, y2 - y1)


def convert_img_to_json_name(img_file_name):
	for ext in ['jpg', 'png']:
		img_file_name = img_file_name.replace('.' + ext, '.json')
	return img_file_name
	

def to_one_hot(input_values, num_classes):
	input_values = np.array(input_values)
	return np.squeeze(np.eye(num_classes)[input_values.reshape(-1)])


def filter_file_paths_from_path_lists(f_paths, lst_paths):
	logTracker.log('Filtering ' + str(len(f_paths)) + ' file path(s) from ' + str(len(lst_paths)) + ' .lst file(s)')	
	selected = None
	if len(lst_paths) > 0:
		selected = set()
		for lst_path in lst_paths:
			temp = set(load_path_list(lst_path))
			selected.update(temp)
			logTracker.log(' - Filtering ' + str(len(temp)) + ' files from ' + lst_path)
	if not selected is None:
		temp = [f_path for f_path in f_paths if getBasename(f_path) in selected]	
	else:
		temp = list(f_paths)
	logTracker.log('Remaining ' + str(len(temp)) + ' file paths')
	return temp


def get_json_file_map_from_file_paths(f_paths):
	res = {}
	f_path_map = {}
	for f_path in f_paths:
		f_name = getBasename(f_path)
		if f_name in res:
			logTracker.logException('Duplicated json file name from "' + str(f_path_map[f_name]) + '" and "' + f_path + '"')
		res[f_name] = load_json(f_path)
		f_path_map[f_name] = f_path
	return res


def get_json_file_map(dir_path):
	f_paths = loadValidFiles(dir_path, 'json', keepFilePath=True)
	return get_json_file_map_from_file_paths(f_paths)


def get_region_size(textline):
	points = textline['location']
	xs = [p[0] for p in points]
	ys = [p[1] for p in points]
	return max(xs) - min(xs), max(ys) - min(ys)


def get_classes_sample_database(sample):
	pts = []
	for textline in sample:
		pts.extend(textline['location'])
	xs = [p[0] for p in pts]
	ys = [p[1] for p in pts]
	sample_w, sample_h = max(xs), max(ys)

	res = defaultdict(lambda: defaultdict(lambda: set()))
	for textline in sample:
		content = textline['text'].strip('\n\t\r')
		if len(content) > 0:
			label_info = textline['label_info']
			fk = label_info['formal_key']
			cur_type = label_info['key_type']
			size = get_region_size(textline)
			res[cur_type][fk].add(frozendict({
				'text'        : content, 
				'size'        : size, 
				'sample_size' : (sample_w, sample_h)
			}))

	return dict(res)


def get_classes_database(samples):
	from chi_lib.ProgressBar import ProgressBar
	logTracker.log('Getting classes database from ' + str(len(samples)) + ' samples')
	res = defaultdict(lambda: defaultdict(lambda: set()))
	progress = ProgressBar(name='Loading', maxValue=len(samples))
	for sample in samples:
		temp = get_classes_sample_database(sample)
		for fk_type, fk_data in temp.items():
			for fk, data in fk_data.items():
				res[fk_type][fk].update(temp[fk_type][fk])
		progress.increase()
	progress.done()
	for fk_type, fk_data in res.items():
		for fk, data in fk_data.items(): 
			res[fk_type][fk] = list(res[fk_type][fk])
	return dict(res)


def load_json(f_path, ignored_not_exists=False):
	if ignored_not_exists is True and not os.path.exists(f_path):
		logTracker.log('Warning: Not existing json file from ' + f_path)
		return None
	with open(f_path, 'r', encoding='utf8') as f:
	# with open(f_path, 'r', encoding='utf-8-sig') as f:
		return json.loads(f.read())
	return None


def save_json(data, f_path, print_status=True):
	if print_status:
		logTracker.log('Saving JSON file to ' + f_path)
	with open(f_path, 'w+', encoding='utf8') as f:
		f.write(json.dumps(data, ensure_ascii=False, indent=4))


def load_jsons(f_paths, ignored_not_exists=False):
	from chi_lib.ProgressBar import ProgressBar
	res = []
	logTracker.log('Loading ' + str(len(f_paths)) + ' json files')
	progress = ProgressBar(name='Loading', maxValue=len(f_paths))
	for f_path in f_paths:
		res.append(load_json(f_path, ignored_not_exists))
		progress.increase()
	progress.done()
	return res


def load_path_list(lstFilePath):
	res = []
	with open(lstFilePath, 'r', encoding='utf8') as f:
		for line in f.readlines():
			if len(line) > 0:
				line = line.strip('\n\t\r ')
				if len(line) > 0:
					res.append(line)	
	return res


def copy_files(f_paths, res_dir_path):
	from chi_lib.ProgressBar import ProgressBar
	logTracker.log('Copying ' + str(len(f_paths)) + ' files to ' + res_dir_path)
	createDirectory(res_dir_path)
	progress = ProgressBar(name='Copying', maxValue=len(f_paths))
	for f_path in f_paths:
		shutil.copy(f_path, res_dir_path)
		progress.increase()
	progress.done()


def merge_directory(resDir, dirPath1, dirPath2):
	from chi_lib.ProgressBar import ProgressBar

	fPaths = [os.path.join(dirPath1, fName) for fName in loadValidFiles(dirPath1, 'json')]
	fPaths += [os.path.join(dirPath2, fName) for fName in loadValidFiles(dirPath2, 'json')]
	createDirectory(resDir)
	logTracker.log('Merging directory:')
	logTracker.log(' - Directory 1 : ' + dirPath1)
	logTracker.log(' - Directory 2 : ' + dirPath2)
	logTracker.log(' - Export to   : ' + resDir)
	logTracker.log('Merging ' + str(len(fPaths)) + ' files')

	progress = ProgressBar(name='Merging', maxValue=len(fPaths))
	for fPath in fPaths:
		shutil.copy(fPath, resDir)
		progress.increase()
	progress.done()
	#shutil.copytree(dirPath2, resDir)
	

def getSubPaths(curPath):
	return [os.path.join(curPath, fName) for fName in loadValidDirectories(curPath)]

def getParentPath(fPath):
	return os.path.dirname(removeEndSep(fPath))	

def getCurrentUsedMemoryPercentage():
	return psutil.virtual_memory().percent

def getCurrentSystemInfo():
	return {
		'cpu' : psutil.cpu_percent(),
		'vm'  : psutil.virtual_memory()
	}

def getLineSeparator(textString, style): 
	if len(textString) == 0:
		return ''
	vals = [len(s) for s in textString.split('\n')]
	maxWidth = np.max(np.array(vals))
	return mergeList(['-' for _ in range(maxWidth)], '')

def getBaseNamesMap(fPaths):
	from chi_lib.ProgressBar import ProgressBar
	res = {}
	logTracker.log('Loading basename map from ' + str(len(fPaths)) + ' paths')
	progress = ProgressBar(name='Loading', maxValue=len(fPaths))
	for fPath in fPaths:
		fName = getBasename(fPath)
		if fName in res:
			postfix = '\n  -' + fPath + '\n  -' + res[fPath]
			logTracker.logException('Duplicated name ' + str(fName) + ' from:' + postfix)
		res[fName] = fPath
		progress.increase()
	progress.done()
	return res


def getTerminalSize():
	# Source: https://stackoverflow.com/questions/566746/how-to-get-linux-console-window-width-in-python/566752#566752
	import os
	env = os.environ
	def ioctl_GWINSZ(fd):
		try:
			import fcntl, termios, struct, os
			cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
		'1234'))
		except:
			return
		return cr
	cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
	if not cr:
		try:
			fd = os.open(os.ctermid(), os.O_RDONLY)
			cr = ioctl_GWINSZ(fd)
			os.close(fd)
		except:
			pass
	if not cr:
		cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

		### Use get(key[, default]) instead of a try/catch
		#try:
		#    cr = (env['LINES'], env['COLUMNS'])
		#except:
		#    cr = (25, 80)
	return int(cr[1]), int(cr[0])

def removeEndSep(curStr):
	curStr = curStr.strip(' ').strip('\n').strip('\r')
	if len(curStr) == 0:
		return ''
	while len(curStr) > 0 and curStr[len(curStr) - 1] == os.path.sep:
		curStr = curStr[:len(curStr) - 1]	
	return curStr

def getBasename(fPath):
	return os.path.basename(removeEndSep(fPath))

def asking(message, answers, delimiter=None, emptyAns=False):
	answers = [str(a) for a in answers]
	isOk = False
	temp = []
	while isOk == False:
		ans = raw_input(message + ': ')
		ans = ans.strip(' ').strip('\t').strip('\n')
		if emptyAns == True and len(ans) == 0:
			return []
		if delimiter != None:
			temp = ans.split(delimiter)
		else:
			temp = [ans]
		isOk = True
		for s in temp:
			s = s.strip(' ').strip('\t') 
			if not 	s in answers:
				logTracker.log("'" + str(s) + "' is not a valid answer")
				isOk = False
				break
	if delimiter == None:
		return temp[0]
	return temp


def loadCSV(fPath):
	logTracker.log('Loading CSV data from ' + str(fPath))
	with open(fPath) as f:
		return [row for row in csv.DictReader(f)]
	logTracker.logException('Cannot open file ' + str(fPath))

def loadValidDirectories(curDir, keepFilePath=False):
	temp = [fName for fName in os.listdir(curDir) if not fName.startswith('.') and os.path.isdir(os.path.join(curDir, fName))]
	if keepFilePath is True:
		temp = [os.path.join(curDir, fName) for fName in temp]
	logTracker.log('Loaded ' + str(len(temp)) + ' directories from ' + curDir)
	return temp

def loadValidFiles(curDir, fType, keepFilePath=False):
	res = None
	if type(fType) is str:
		res = [fName for fName in os.listdir(curDir) if not fName.startswith('.') and os.path.isfile(os.path.join(curDir, fName)) and fName.endswith('.' + str(fType))]	
	elif type(fType) is list or type(fType) is set:
		res = []
		for fName in os.listdir(curDir):
			if not fName.startswith('.') and os.path.isfile(os.path.join(curDir, fName)):
				for t in fType:
					if fName.endswith('.' + str(t)):
						res.append(fName)
						break
	if res is None:
		logTracker.logException('Invalid fType: ' + str(fType))
	if keepFilePath is True:
		res = [os.path.join(curDir, fName) for fName in res]

	logTracker.log('Loaded ' + str(len(res)) + ' "' + str(fType) + '" files from ' + curDir)
	return res


def isEqualWithKeys(dict1, dict2, keys):
	for k in keys:
		if dict1[k] != dict2[k]:
			return False
	return True

def splitList(valuesList, batchSize):
	res = []
	temp = []
	for i in range(len(valuesList)):
		if len(temp) < batchSize:
			temp.append(valuesList[i])
		else:
			res.append(temp)
			temp = [valuesList[i]]
	if len(temp) > 0:
		res.append(temp)
	return res

def quantileList(valuesList, n):
	res = []
	blockSize = len(valuesList) / n
	remaining = len(valuesList) % n
	startIdx = 0
	endIdx = blockSize + min(1, remaining)
	for i in range(n):
		res.append(valuesList[startIdx:endIdx])
		remaining = max(0, remaining - 1)
		startIdx = endIdx
		endIdx += blockSize + min(1, remaining)
	return res	

def mergeList(valuesList, delimiter):
	if len(valuesList) == 0:
		return ''
	res = ''
	for i in range(len(valuesList) - 1):
		res += str(valuesList[i]) + str(delimiter)
	return res + str(valuesList[len(valuesList) - 1])

def isEmptyDirectory(dirPath):
	fNames = [fName for fName in os.listdir(dirPath) if not fName.startswith('.') and not fName.endswith('.')]
	if len(fNames) == 0:
		return True
	return False

def productDict(curDict):
	keys = curDict.keys()
	vals = curDict.values()
	res = []
	for instance in itertools.product(*vals):
		res.append(dict(zip(keys, instance)))
	return res

def isJson(rawText):
	try:
		json.loads(rawText)
	except ValueError:
		return False
	return True

def isAllInMap(valuesMap, keysList):
	for k in keysList:
		if not k in valuesMap:
			return False
	return True	

def createDirectory(dirPath):
	if not os.path.isdir(dirPath):
		os.makedirs(dirPath)

def writeData(fPath, values):
	with open(fPath, 'w+') as f:
		for v in values:
			f.write(v + '\n')
		f.close()

def loadFields(fPath):
	res = []
	with open(fPath) as f:
		for line in f.readlines():
			res.append(line.strip())
		f.close()
	return res

def getDatetimeFromString(timeStr, dateFormat):
	return dt.strptime(timeStr, dateFormat)

def getStringFromDatetime(dateTime, dateFormat):
	return dateTime.strftime(dateFormat)	

def getTodayDatetimeString(dateFormat='%Y%m%d-%H%M%S'):
	return getStringFromDatetime(dt.now(), dateFormat)


