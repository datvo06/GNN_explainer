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
    max_x = np.max(list_bboxs[:, 0] + list_bboxs[:, 2] - 1)
    max_y = np.max(list_bboxs[:, 1] + list_bboxs[:, 3] - 1)
    list_bboxs[:, 0] = (list_bboxs[:, 0] - min_x + clamp_min)/(
        max_x - min_x + clamp_min)

    list_bboxs[:, 1] = list_bboxs[:, 1] - min_y + clamp_min/(
        max_x - min_x + clamp_min)

    list_bboxs[:, 2] = (list_bboxs[:, 2] + clamp_min)/(
        max_x - min_x + clamp_min)
    list_bboxs[:, 3] = (list_bboxs[:, 3] + clamp_min)/(
        max_y - min_y + clamp_min)
    return list_bboxs


if __name__ == '__main__':
    data_preprocessed_list = []
    for json_filename in glob.glob(
            "dataset/training_data/annotations/*.json"):
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
    pickle.dump(data_preprocessed_list,
                open('funsd_preprocess.pkl', 'wb'))
