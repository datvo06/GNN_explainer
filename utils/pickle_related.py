import pickle, json


def read_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        info = pickle.load(f)
    return info


def load_json(f_path):
    # with open(f_path, 'r', encoding='utf-8-sig') as f:
    with open(f_path, 'r', encoding='utf8') as f:
        return json.loads(f.read())
