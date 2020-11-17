from __future__ import print_function, unicode_literals
from kv.graphkv_torch.utils.kv_ca_dataset import KV_CA_Dataset
from kv.graphkv_torch.utils.data_encoder import (
    InputEncoder, convert_label_to_cassia)

import os
import sys
import json
import numpy as np
import pickle as pickle
import glob
import torch



def representInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def convert_label_via_to_cassia(label_data: dict,
                                use_clusters: bool = False) -> (list, list):
    """
        Cassia Input Format is list with each region is a dict that map:
            - "location": Four point coordinate in clockwise position
            - "text": Text data
        Return 2 list,
            - list of tuple class_name and key_type from label
            - list of Cassia format for input
    """

    cassia_input = []
    list_label = []
    regions = label_data['regions']

    for region in regions:
        shape = region['shape_attributes']
        if shape['name'] == 'polygon':
            xs = shape['all_points_x']
            ys = shape['all_points_y']
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            location = [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ]
        elif shape['name'] == 'rect':
            x, y = shape['x'], shape['y']
            w, h = shape['width'], shape['height']

            location = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ]
        else:
            warnings.warn(f"Not supported {shape['name']}")

        text = region['region_attributes']['label']
        class_name = region['region_attributes']['formal_key']
        if use_clusters:
            # Split it further
            cluster_name = int(
                class_name.split('_')[-1])\
                if representInt(class_name.split('_')[-1]) else 0

        key_type = region['region_attributes']["key_type"]

        cassia_input.append({
            "location": location,
            "text": text
        })

        if use_clusters:
            list_label.append((cluster_name, class_name, key_type))
        else:
            list_label.append((class_name, key_type))

    return list_label, cassia_input


class KV_CA_Dataset_modified(KV_CA_Dataset):
    def __init__(self,
                 label_paths: list,
                 encoder,
                 classes: list,
                 clusters: list,
                 key_types: list,
                 take_original_input=False
                 ):
        """ Dataset class to load CA Format file
        Params:
            label_paths (list of path): List of path to labels folder CA Labels
            encoder: Function that take cassia kv_input
                and convert to vertex
            classes: List of classes (Remeber classes[0] for field that we
            dont care)
            key_types: List of key type you want to train
        Optional:
            graph_builder: Function that take cassia kv_input
                and convert to adjacency matrix
            num_graph_channel: The channel of Number of Graph,
            only support last now
        """
        super().__init__(
            label_paths,
            encoder,
            classes,
            clusters=None,
            key_types=['key', 'value']
        )

        self.take_original_input = take_original_input

    def convert_label_to_numerical(self, labels):
        # Convert labels to index
        if not self.use_clusters:
            target = []
            for class_name, key_type in labels:
                class_index = self.cls2ind.get(class_name, {}).get(key_type, 0)
                target.append(class_index)
            return torch.tensor(target)
        else:
            target = []
            target_cluster = []
            for cluster_name, class_name, key_type in labels:
                class_index = self.cls2ind.get(class_name, {}).get(key_type, 0)
                cluster_index = self.cluster2ind.get(
                    cluster_name, {}).get(key_type, 0)
                target.append(class_index)
                target_cluster.append(cluster_index)
            return torch.tensor(target), torch.tensor(target_cluster)

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        label_data = json.load(open(label_path, encoding='utf-8'))

        if not self.use_cinnamon_format:
            labels, kv_input = convert_label_via_to_cassia(
                label_data,
                use_clusters=self.use_clusters
            )

            # Convert labels to numerical targets
            if not self.use_clusters:
                target = self.convert_label_to_numerical(labels)
            else:
                target, target_cluster = self.convert_label_to_numerical(labels)
        else:
            labels, kv_input = convert_label_to_cassia(label_data)
            target = self.convert_label_to_numerical(labels)

        # Convert kv_input to numerical input
        vertex, adj_matrix = self.encode_kv_input(kv_input)

        if not self.use_clusters:
            if self.take_original_input:
                return vertex, adj_matrix, target, kv_input
            return vertex, adj_matrix, target
        else:
            if self.take_original_input:
                return vertex, adj_matrix, target, target_cluster, kv_input
            return vertex, adj_matrix, target, target_cluster

    def __len__(self):
        return self.num_data


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
        dataset = KV_CA_Dataset_modified(all_files,
                InputEncoderKeepKVData(json.load(open(sys.argv[2], 'r'))),
                json.load(open(sys.argv[3])),
                clusters=None,
                key_types=['key', 'value'],
                take_original_input=True
                )
    except:
        dataset = KV_CA_Dataset_modified(all_files,
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
