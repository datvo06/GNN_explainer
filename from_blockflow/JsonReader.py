import os, sys, json
import time
import shutil
import numpy as np
import tensorflow as tf
sys.path.append('../datahub/') 

from normalizing.normalize_text import normalize_text
from copy import deepcopy
from chi_lib.ProgressBar import ProgressBar
from chi_lib.library import *

from blocks.Block import Block


class JsonReader(Block):
	def __init__(self, json_paths, name=None):
		Block.__init__(self, [], name=name)
		self.json_paths = list(json_paths)

	def implement(self):
		return {
			'json_samples' : (None, load_jsons(self.json_paths)),
			'json_file_paths' : (None, list(self.json_paths))
		}