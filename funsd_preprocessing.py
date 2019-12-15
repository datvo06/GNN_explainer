from __future__ import print_function, unicode_literals
import json
import glob
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import utils.graph_building_utils as gbutils
import pickle

transformer_model = SentenceTransformer('bert-base-nli-mean-tokens')

__author__ = "Marc"


def normalize_pos_feats(list_bboxs, clamp_min=0.1):
    list_bboxs = np.array(list_bboxs).astype('float') # expected to be Nx4
    min_x = np.min(list_bboxs[:, 0])
    min_y = np.min(list_bboxs[:, 1])
    max_x = np.max(list_bboxs[:, 0] + list_bboxs[:, 2])
    max_y = np.max(list_bboxs[:, 1] + list_bboxs[:, 3])
    list_bboxs[:, 0] = (list_bboxs[:, 0] - min_x)/(
        max_x - min_x)

    list_bboxs[:, 1] = (list_bboxs[:, 1] - min_y)/(
        max_x - min_x)

    list_bboxs[:, 2] = (list_bboxs[:, 2])/(
        max_x - min_x)
    list_bboxs[:, 3] = (list_bboxs[:, 3])/(max_y - min_y)
    list_bboxs = (list_bboxs + clamp_min)/(clamp_min+1.0)
    return list_bboxs


def get_preprocessed_list(dirpath):
    data_preprocessed_list = []
    for json_filename in glob.glob(os.path.join(dirpath, "*.json")):
        list_bboxs = []
        list_ocrs = []
        list_labels = []
        list_linking = []
        list_ids = []
        json_dict = json.load(open(json_filename))['form']
        for textline in json_dict:
            list_bboxs.append([textline['box'][0],
                               textline['box'][1],
                               textline['box'][2]-textline['box'][0]+1,
                               textline['box'][3]-textline['box'][1]+1])
            list_ocrs.append(textline['text'])
            list_labels.append(textline['label'])
            list_linking.append(textline['linking'])
            list_ids.append(textline['id'])
        cell_list = gbutils.get_list_cells(list_bboxs, list_ocrs)
        adj_mats = gbutils.get_adj_mat(cell_list)
        feats = transformer_model.encode(list_ocrs)
        data_dict = {'file_path': json_filename}
        data_dict['cells'] = cell_list
        data_dict['adj_mats'] = adj_mats
        data_dict['pos_feats'] = normalize_pos_feats(list_bboxs)
        data_dict['transformer_feature'] = np.array(feats)
        data_dict['labels'] = list_labels
        data_dict['link'] = list_linking
        data_preprocessed_list.append(data_dict)
    return data_preprocessed_list

if __name__ == '__main__':
    pickle.dump(get_preprocessed_list(
        "dataset/training_data/annotations/"),
        open('funsd_preprocess_train.pkl', 'wb'))
    pickle.dump(get_preprocessed_list(
        "dataset/testing_data/annotations/"),
        open('funsd_preprocess_test.pkl', 'wb'))
