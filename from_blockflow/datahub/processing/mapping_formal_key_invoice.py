# encoding: utf-8


import os, sys, json
import glob                                                           
import cv2 
import numpy as np
import argparse
import shutil

from chi_lib.library import *
from collections import defaultdict
from frozendict import frozendict
from chi_lib.Multiprocessor import Multiprocessor
from chi_lib.ProgressBar import ProgressBar
from directory_name import *


def mapping(json_paths, required_keys, key_map, res_dir):
	createDirectory(res_dir)
	key_map = {k.lower() : [kk.lower() for kk in k_list] for k, k_list in key_map.items()}	
	
	logTracker.log('Mapping formal key for ' + str(len(json_paths)) + ' labeled files. Result at ' + res_dir)
	progress = ProgressBar(name='Converting', maxValue=len(json_paths))
	for fPath in json_paths:
		progress.increase()
		with open(fPath, 'rb') as f:
			textlines = json.loads(f.read())

			# ------------- Invoice project special part is here -------------
			existingKeys = set()
			for textline in textlines:
				fk = textline['label_info']['formal_key'].lower()
				existingKeys.add(fk)
			if 'item_total_including_tax' in existingKeys:
				if 'item_total_excluding_tax' in existingKeys:
					existingKeys.remove('item_total_including_tax')

			for textline in textlines:
				label_info = textline['label_info']
				fk = label_info['formal_key'].lower()
				if not fk in existingKeys:
					label_info['formal_key'] = 'None'

			# ----------------------------------------------------------------
			
			for textline in textlines:
				label_info = textline['label_info']
				new_fk = label_info['formal_key']
				fk = new_fk.lower()
				if fk in key_map:
					isExisting = False
					for kk in key_map[fk]:
						if kk in required_keys:
							new_fk = kk
							isExisting = True
							break

					if not isExisting:
						new_fk = 'None'

				if new_fk != 'None':
					new_fk = new_fk.lower()
				if not new_fk in required_keys:
					new_fk = 'None'
				label_info['formal_key'] = new_fk

				cur_fk_type = label_info['key_type'].lower()
				label_info['key_type'] = str(cur_fk_type) if cur_fk_type in ['key', 'value'] else 'other'

		resPath = os.path.join(res_dir, getBasename(fPath))
		with open(resPath, 'w+', encoding='utf8') as f:
			f.write(json.dumps(textlines, ensure_ascii=False, indent=4))
	progress.done()