import pandas as pd
import numpy as np
import json
import pickle
import os

def convert_to_df(data_object):
    if isinstance(data_object, pd.DataFrame):
        return data_object
    if isinstance(data_object, list):
        try:
            return pd.DataFrame.from_records(data_object)
        except:
            raise ValueError('Data object appears to be a list, but it can\'t be parsed by pandas.DataFrame.from_records(...)')
    if isinstance(data_object, dict):
        try:
            return pd.DataFrame.from_dict(data_object, orient='index')
        except:
            raise ValueError('Data object appears to be a dict, but it can\'t be parsed by pandas.DataFrame.from_dict(..., orient=\'index\')')
    if isinstance(data_object, str) and os.path.isfile(data_object):
        file_extension = os.path.splitext(data_object)[1]
        if len(file_extension) == 0:
            raise ValueError('Data object appears to be a filename, but it has no extension')
        if file_extension.lower() == '.pickle' or file_extension.lower() == '.pkl':
            try:
                data_deserialized = pickle.load(open(data_object, 'rb'))
            except:
                raise ValueError('File has extension \'pickle\' or \'pkl\', but pickle deserialization failed')
            return convert_to_df(data_deserialized)
        if file_extension.lower() == '.json':
            try:
                data_deserialized = json.load(open(data_object, 'r'))
            except:
                raise ValueError('File has extension \'json\', but JSON deserialization failed')
            return convert_to_df(data_deserialized)
        if file_extension.lower() == '.csv':
            try:
                data_deserialized = pd.read_csv(data_object)
            except:
                raise ValueError('File has extension \'csv\', but pd.read_csv(...) failed')
            return data_deserialized
        raise ValueError('Data object appears to be a filename, but extension \'{}\' isn\'t currently handled'.format(
            file_extension
        ))
    if isinstance(data_object, str):
            try:
                data_deserialized = json.loads(data_object)
            except:
                raise ValueError('Data object is a string but it doesn\'t appear to be a valid filename or valid JSON')
            return convert_to_df(data_deserialized)
    raise ValueError('Failed to parse data object')
