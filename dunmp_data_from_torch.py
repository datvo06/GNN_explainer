from __future__ import print_function, unicode_literals
from kv.graphkv_torch.utils.kv_ca_dataset import KV_CA_Dataset
from kv.graphkv_torch.utils.data_encoder import InputEncoder

import os
import sys
import json
import numpy as np
import pickle as pickle
import glob


class InputEncoderKeepKVData(InputEncoder):
    """ The same as the parent, but also keep the text
    in short, the output =\
        ({"location": location feature, "text": text}, ... - InputEncoder data)
    Example:
        encoder = InputEncoder(corpus)
        vertex_data = encoder(kv_input)
    """
    def __init__(self, corpus, use_cache_text=True, is_normalized_text=False):
        super().__init__(
            corpus, use_cache_text,
            is_normalized_text)

    def encode(self, sample, normalize_vertex_func=None):
        encoded_data = super().encode(
            sample, normalize_vertex_func)
        return (sample, encoded_data)

    def __call__(self, sample, normalize_vertex_func=None):
        # Use call warper for more pytorch like
        return self.encode(sample, normalize_vertex_func)


if __name__ == '__main__':
    # first arg: path to directory containing all json lables
    # second args: corpus.json path
    # third path: class.json path
    all_files = list(glob.glob(os.path.join(sys.argv[1], '*.json')))
    try:
        dataset = KV_CA_Dataset(all_files,
                InputEncoderKeepKVData(json.load(open(sys.argv[2], 'r'))),
                json.load(open(sys.argv[3])),
                clusters=None,
                key_types=['key', 'value'],
                take_original_input=True
                )
    except:
        dataset = KV_CA_Dataset(all_files,
                InputEncoderKeepKVData(json.load(open(sys.argv[2], 'r'))),
                json.load(open(sys.argv[3])),
                key_types=['key', 'value'],
                take_original_input=True
                )

    output_dict = {
            'file_paths': [],
            'location': [],
            'text': [],
            'original_location': [],
            'HeuristicGraphAdjMat':[],
            'FormBowFeature': [],
            'TextlineCoordinateFeature': [],
            'labels': []
    }
    for i in range(len(dataset)):

        encoded_data, adj_matrix, target, _ = dataset[i]
        kv_input, vertex = encoded_data
        vertex = vertex.cpu().detach().numpy()

        bow_features = vertex[:, :-4]
        location_features = vertex[:, -4:]

        adj_matrix = adj_matrix.cpu().detach().numpy()
        labels = target.cpu().detach().numpy()

        output_dict['file_paths'].append(all_files[i])
        output_dict['text'].append([box['text'] for box in kv_input])
        output_dict['location'].append([ box ['location'] for box in kv_input])
        output_dict['HeuristicGraphAdjMat'].append(adj_matrix)
        output_dict['FormBowFeature'].append(bow_features)
        output_dict['TextlineCoordinateFeature'].append(location_features)
        output_dict['labels'].append(labels)
    pickle.dump(output_dict, open('input_features.pickle', 'wb'))
