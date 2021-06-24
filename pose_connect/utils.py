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

def set_index_columns(
    df,
    index_columns
):
    if df.index.name == index_columns or df.index.names == index_columns:
        return df
    df = df.copy()
    if isinstance(index_columns, str):
        if df.index.nlevels == 1 and df.index.name is None and index_columns not in df.columns:
            df.index.name = index_columns
            return df
        if (df.index.nlevels == 1 and df.index.name is None) or (df.index.nlevels > 1 and df.index.names == [None]*df.index.nlevels):
            df.reset_index(inplace=True, drop=True)
        else:
            df.reset_index(inplace=True)
        if index_columns not in df.columns:
            raise ValueError('Dataframe already had a named index and specified index name not in column names')
        df.set_index(index_columns, inplace=True)
        return df
    else:
        try:
            num_target_levels = len(index_columns)
        except:
            raise ValueError('Specified index columns must either be string or sequence of strings')
        if num_target_levels == 1 and df.index.nlevels == 1 and df.index.name is None and not set(index_columns).issubset(set(df.columns)):
            df.index.name = index_columns[0]
            return df
        if num_target_levels > 1 and df.index.nlevels == num_target_levels and df.index.names == [None]*df.index.nlevels and not set(index_columns).issubset(df.columns):
            df.index.names = index_columns
            return df
        if (df.index.nlevels == 1 and df.index.name is None) or (df.index.nlevels > 1 and df.index.names is [None]*df.index.nlevels):
            df.reset_index(inplace=True, drop=True)
        else:
            df.reset_index(inplace=True)
        if not set(index_columns).issubset(set(df.columns)):
            raise ValueError('Dataframe already had a named index and specified index names not in column names')
        df.set_index(index_columns, inplace=True)
        return df
