from __future__ import print_function, unicode_literals
from kv.graphkv_torch.utils.kv_ca_dataset import KV_CA_Dataset
from kv.graphkv_torch.utils.data_encoder import InputEncoder

import os
import sys
import json
import numpy as np
import pickle as pickle
import glob


if __name__ == '__main__':
    # first arg: path to directory containing all json lables
    # second args: corpus.json path
    # third path: class.json path
    all_files = list(glob.glob(os.path.join(sys.argv[1], '*.json')))
    try:
        dataset = KV_CA_Dataset(all_files,
                InputEncoder(json.load(open(sys.argv[2], 'r'))),
                json.load(open(sys.argv[3])),
                clusters=None,
                key_types=['key', 'value'],
                take_original_input=True
                )
    except:
        dataset = KV_CA_Dataset(all_files,
                InputEncoder(json.load(open(sys.argv[2], 'r'))),
                json.load(open(sys.argv[3])),
                key_types=['key', 'value'],
                take_original_input=True
                )
ataset = KV_CA_Dataset(all_files,
                InputEncoder(json.load(open(sys.argv[2], 'r'))),
                json.load(open(sys.argv[3])),
                clusters=None,
                key_types=['key', 'value']
                )


    output_dict = {
            'file_paths': [],
            'HeuristicGraphAdjMat':[],
            'FormBowFeature': [],
            'TextlineCoordinateFeature': [],
            'labels': []
    }
    for i in range(len(dataset)):
        vertex, adj_matrix, target = dataset[i]
        vertex = vertex.cpu().detach().numpy()

        bow_features = vertex[:, :-4]
        location_features = vertex[:, -4:]

        adj_matrix = adj_matrix.cpu().detach().numpy()
        labels = target.cpu().detach().numpy()

        output_dict['file_paths'].append(all_files[i])
        output_dict['HeuristicGraphAdjMat'].append(adj_matrix)
        output_dict['FormBowFeature'].append(bow_features)
        output_dict['TextlineCoordinateFeature'].append(location_features)
        output_dict['labels'].append(labels)
    pickle.dump(output_dict, open('input_features.pickle', 'wb'))
