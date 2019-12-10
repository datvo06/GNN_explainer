import os, cv2
import numpy as np
from utils.pickle_related import read_pickle, load_json
from utils.draw_utils import visualize_graph


def preprocess_class(json_path, class_types=['key', 'value']):
    """
    :param json_path: path of classes.json
    :param class_types: What types do we have for per class in classes.json.
    :return: list of extended classes, where
             len(out) = dim(model output)
    """

    out = load_json(json_path)
    out = [c + "_" + ct for c in out for ct in class_types]
    out = ["None"] + out

    return out


if __name__ == '__main__':
    features_folder = "../Invoice_k_fold/save_features/all/"
    pickle_path = "input_features.pickle"
    class_path = "classes.json"
    bow_path = "corpus.json"

    # dict_keys(['file_paths', 'HeuristicGraphAdjMat', 'FormBowFeature', 'TextlineCoordinateFeature', 'labels'])
    input_data = read_pickle(
        os.path.join(features_folder, pickle_path)
    )
    corpus = open(os.path.join(features_folder, bow_path)).read()[1:-2]
    class_kv = preprocess_class(os.path.join(features_folder, class_path))

    i = 0
    adj = input_data['HeuristicGraphAdjMat'][i]
    adj = np.transpose(adj, (0, 2, 1))
    bow = input_data['FormBowFeature'][i]
    coord = input_data['TextlineCoordinateFeature'][i]
    coord = coord*1.1 - 0.1
    label_y = input_data['labels'][i]
    N = bow.shape[0]

    image = visualize_graph(list_bows=bow,
                            list_positions=(coord*1000).astype(int),
                            adj_mats=adj,
                            node_labels=label_y,
                            node_importances=np.random.random(N),
                            # node_importances=N*[4*[1]],
                            position_importances=np.random.random((N, 4, 1)),
                            # position_importances=N*[1],
                            # bow_importances=N*[bow.shape[-1] * [1]],
                            bow_importances=np.random.random((N, bow.shape[-1])),
                            adj_importances=np.random.random(adj.shape),
                            word_list=corpus
                            )

    cv2.imshow('My Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



'''

visualize_graph(list_bows, list_positions,
                    adj_mats, node_labels,
                    node_importances,
                    position_importances,
                    bow_importances,
                    adj_importances,
                    orig_img=None,
                    word_list=None, is_text=False)

'''


