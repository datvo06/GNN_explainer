import os, sys, json, re
import numpy as np
import tensorflow as tf

sys.path.append('../datahub/')

from normalizing.normalize_text import normalize_text
from utils.utils import load_frozen_graph
from chi_lib.library import *
from utils.utils import *
from chi_lib.ProgressBar import ProgressBar
from blocks.Block import Block


class OneToOneTextlineInferrer(Block):
    def __init__(self, predictions_inferrer_block, fileloader_block, classes, corpus, name=None):
        Block.__init__(self, [predictions_inferrer_block, fileloader_block], name=name)
        self.predictions_inferrer_block = predictions_inferrer_block
        self.fileloader_block = fileloader_block
        self.classes = list(classes)
        self.corpus = corpus

    def implement(self):
        return {}

    def execute(self):
        predictions = self.predictions_inferrer_block.get_latest_executed()['predictions']
        _, json_samples = self.fileloader_block.get()['json_samples']
        _, json_file_paths = self.fileloader_block.get()['json_file_paths']

        mask = self.get_mask()
        json_samples = filter_by_mask(json_samples, mask)
        json_file_paths = filter_by_mask(json_file_paths, mask)

        if len(json_samples) != len(predictions) or len(json_file_paths) != len(predictions):
            logTracker.logException(
                'Inconsistent number of samples and predictions: {} - {} - {}'.format(len(json_samples),
                                                                                      len(predictions),
                                                                                      len(json_file_paths)))

        logTracker.log('Inferring textline for ' + str(len(json_samples)) + ' samples')

        outputs = []
        corpus_cleaner = re.compile(r'[^' + re.escape(self.corpus) + r']')
        progress = ProgressBar('Inferring', len(json_samples))
        for i, sample in enumerate(json_samples):
            cur_prediction = predictions[i].tolist()
            cur_input = deepcopy(sample)
            if len(cur_prediction) != len(sample):
                logTracker.logException(
                    'Inconsistent number of textlines and textline predictions: {} - {}'.format(len(cur_prediction),
                                                                                                len(sample)))

            for j, textline in enumerate(sample):
                cur_input[j]['type'] = self.classes[cur_prediction[j]]
                # cur_input[j]['confidence'] = round(float(probs[j]), 4)
                norm_text = normalize_text(cur_input[j]['text'])
                norm_text = corpus_cleaner.sub('�', norm_text)
                cur_input[j]['infer_info'] = {
                    'norm_text': norm_text,
                    'bow': ''.join(sorted(set(norm_text)))
                }
            outputs.append(cur_input)
            progress.increase()
        progress.done()
        outputs = list(zip(json_file_paths, outputs))
        return {
            'json_outputs': outputs
        }


