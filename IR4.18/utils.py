import pickle
import os


def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_cache_filename(prefix, feat_type, n_clusters, des_type=None):
    if des_type:
        return f"{prefix}_{feat_type}_{n_clusters}_{des_type}.pkl"
    else:
        return f"{prefix}_{feat_type}_{n_clusters}.pkl"