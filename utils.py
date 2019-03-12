import json
import pickle


def save_params(params, file_path):
    pickle.dump(params, open(file_path, 'wb'))


def load_params(params, file_path):
    return pickle.load(open(file_path, 'rb'))

