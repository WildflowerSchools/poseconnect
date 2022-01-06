import pandas as pd
import numpy as np
import json
import pickle
import os

def ingest_poses_generic(
    data_object,
    pose_type
):
    df = convert_to_df(data_object)
    all_column_names = df.reset_index().columns
    if pose_type=='2d':
        df = set_index_columns(
            df=df,
            index_columns='pose_2d_id'
        )
    elif pose_type=='3d':
        df = set_index_columns(
            df=df,
            index_columns='pose_3d_id'
        )
    else:
        raise ValueError('Pose type \'{}\' not recognized'.format(
            pose_type
        ))
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    if 'camera_id' in all_column_names:
        df['camera_id'] = df['camera_id'].astype('object')
    if 'keypoint_coordinates_2d' in all_column_names:
        df['keypoint_coordinates_2d'] = df['keypoint_coordinates_2d'].apply(convert_to_array)
    if 'keypoint_quality_2d' in all_column_names:
        df['keypoint_quality_2d'] = df['keypoint_quality_2d'].apply(convert_to_array)
    if 'pose_quality_2d' in all_column_names:
        df['pose_quality_2d'] = pd.to_numeric(df['pose_quality_2d']).astype('float')
    if 'keypoint_coordinates_3d' in all_column_names:
        df['keypoint_coordinates_3d'] = df['keypoint_coordinates_3d'].apply(convert_to_array)
    if 'pose_2d_ids' in all_column_names:
        df['pose_2d_ids'] = df['pose_2d_ids'].apply(convert_to_list)
    return df

def ingest_poses_2d(data_object):
    df = convert_to_df(data_object)
    df = set_index_columns(
        df=df,
        index_columns='pose_2d_id'
    )
    target_columns = [
        'timestamp',
        'camera_id',
        'keypoint_coordinates_2d',
        'keypoint_quality_2d',
        'pose_quality_2d'
    ]
    if not set(target_columns).issubset(set(df.columns)):
        raise ValueError('Data is missing fields: {}'.format(
            set(target_columns) - set(df.columns)
        ))
    df = df.reindex(columns=target_columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['camera_id'] = df['camera_id'].astype('object')
    df['keypoint_coordinates_2d'] = df['keypoint_coordinates_2d'].apply(convert_to_array)
    df['keypoint_quality_2d'] = df['keypoint_quality_2d'].apply(convert_to_array)
    df['pose_quality_2d'] = pd.to_numeric(df['pose_quality_2d']).astype('float')
    return df

def ingest_camera_calibrations(data_object):
    df = convert_to_df(data_object)
    df = set_index_columns(
        df=df,
        index_columns='camera_id'
    )
    target_columns = [
        'camera_matrix',
        'distortion_coefficients',
        'rotation_vector',
        'translation_vector',
        'image_width',
        'image_height'
    ]
    if not set(target_columns).issubset(set(df.columns)):
        raise ValueError('Data is missing fields: {}'.format(
            set(target_columns) - set(df.columns)
        ))
    df = df.reindex(columns=target_columns)
    df['camera_matrix'] = df['camera_matrix'].apply(convert_to_array)
    df['distortion_coefficients'] = df['distortion_coefficients'].apply(convert_to_array)
    df['rotation_vector'] = df['rotation_vector'].apply(convert_to_array)
    df['translation_vector'] = df['translation_vector'].apply(convert_to_array)
    return df

def ingest_poses_3d(data_object):
    df = convert_to_df(data_object)
    df = set_index_columns(
        df=df,
        index_columns='pose_3d_id'
    )
    target_columns = [
        'timestamp',
        'keypoint_coordinates_3d',
        'pose_2d_ids'
    ]
    if not set(target_columns).issubset(set(df.columns)):
        raise ValueError('Data is missing fields: {}'.format(
            set(target_columns) - set(df.columns)
        ))
    df = df.reindex(columns=target_columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['keypoint_coordinates_3d'] = df['keypoint_coordinates_3d'].apply(convert_to_array)
    df['pose_2d_ids'] = df['pose_2d_ids'].apply(convert_to_list)
    return df

def output_poses_3d(df, path):
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    file_extension = os.path.splitext(path)[1]
    if len(file_extension) == 0:
        raise ValueError('Output path has no extension')
    if file_extension.lower() == '.pickle' or file_extension.lower() == '.pkl':
        try:
            df.to_pickle(path)
        except:
            raise ValueError('Output path has extension \'pickle\' or \'pkl\', but pickle serialization failed')
    elif file_extension.lower() == '.json':
        try:
            df.reset_index().to_json(
                path,
                orient='records',
                date_format='iso',
                indent=2
            )
        except:
            raise ValueError('Output path has extension \'json\', but JSON serialization failed')
    elif file_extension.lower() == '.csv':
        try:
            df['timestamp'] = df['timestamp'].apply(lambda x: x.isoformat())
            df['keypoint_coordinates_3d'] = df['keypoint_coordinates_3d'].apply(lambda x: x.tolist())
            df['pose_2d_ids'] = df['pose_2d_ids'].apply(lambda x: json.dumps(x))
            df.to_csv(path)
        except:
            raise ValueError('Output path has extension \'csv\', but conversion to CSV failed')
    else:
        raise ValueError('Data object appears to be a filename, but extension \'{}\' isn\'t currently handled'.format(
            file_extension
        ))

def ingest_poses_3d_with_tracks(data_object):
    df = convert_to_df(data_object)
    df = set_index_columns(
        df=df,
        index_columns='pose_3d_id'
    )
    target_columns = [
        'timestamp',
        'keypoint_coordinates_3d',
        'pose_2d_ids',
        'pose_track_3d_id'
    ]
    if not set(target_columns).issubset(set(df.columns)):
        raise ValueError('Data is missing fields: {}'.format(
            set(target_columns) - set(df.columns)
        ))
    df = df.reindex(columns=target_columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['keypoint_coordinates_3d'] = df['keypoint_coordinates_3d'].apply(convert_to_array)
    df['pose_2d_ids'] = df['pose_2d_ids'].apply(convert_to_list)
    return df

def output_poses_3d_with_tracks(df, path):
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    file_extension = os.path.splitext(path)[1]
    if len(file_extension) == 0:
        raise ValueError('Output path has no extension')
    if file_extension.lower() == '.pickle' or file_extension.lower() == '.pkl':
        try:
            df.to_pickle(path)
        except:
            raise ValueError('Output path has extension \'pickle\' or \'pkl\', but pickle serialization failed')
    elif file_extension.lower() == '.json':
        try:
            df.reset_index().to_json(
                path,
                orient='records',
                date_format='iso',
                indent=2
            )
        except:
            raise ValueError('Output path has extension \'json\', but JSON serialization failed')
    elif file_extension.lower() == '.csv':
        try:
            df['timestamp'] = df['timestamp'].apply(lambda x: x.isoformat())
            df['keypoint_coordinates_3d'] = df['keypoint_coordinates_3d'].tolist()
            df['pose_2d_ids'] = df['pose_2d_ids'].apply(lambda x: json.dumps(x))
            df.to_csv(path)
        except:
            raise ValueError('Output path has extension \'csv\', but conversion to CSV failed')
    else:
        raise ValueError('Data object appears to be a filename, but extension \'{}\' isn\'t currently handled'.format(
            file_extension
        ))

def ingest_poses_3d_with_person_ids(data_object):
    df = convert_to_df(data_object)
    df = set_index_columns(
        df=df,
        index_columns='pose_3d_id'
    )
    target_columns = [
        'timestamp',
        'keypoint_coordinates_3d',
        'pose_2d_ids',
        'pose_track_3d_id',
        'person_id'
    ]
    if not set(target_columns).issubset(set(df.columns)):
        raise ValueError('Data is missing fields: {}'.format(
            set(target_columns) - set(df.columns)
        ))
    df = df.reindex(columns=target_columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['keypoint_coordinates_3d'] = df['keypoint_coordinates_3d'].apply(convert_to_array)
    df['pose_2d_ids'] = df['pose_2d_ids'].apply(convert_to_list)
    return df

def ingest_sensor_data(
    data_object,
    id_field_names=['person_id']
):
    df = convert_to_df(data_object)
    df = set_index_columns(
        df=df,
        index_columns='position_id'
    )
    target_columns = (
        ['timestamp'] +
        id_field_names +
        [
            'x_position',
            'y_position',
            'z_position'
        ]
    )
    if not set(target_columns).issubset(set(df.columns)):
        raise ValueError('Data is missing fields: {}'.format(
            set(target_columns) - set(df.columns)
        ))
    df = df.reindex(columns=target_columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['x_position'] = pd.to_numeric(df['x_position']).astype('float')
    df['y_position'] = pd.to_numeric(df['y_position']).astype('float')
    df['z_position'] = pd.to_numeric(df['z_position']).astype('float')
    return df

def ingest_sensor_position_keypoint_index(data_object):
    if data_object is None:
        return None
    if isinstance(data_object, int):
        return data_object
    if isinstance(data_object, dict):
        return data_object
    if isinstance(data_object, str) and os.path.isfile(data_object):
        file_extension = os.path.splitext(data_object)[1]
        if len(file_extension) == 0:
            raise ValueError('Data object appears to be a filename, but it has no extension')
        if file_extension.lower() == '.pickle' or file_extension.lower() == '.pkl':
            try:
                data_deserialized = pickle.load(open(data_object, 'rb'))
            except:
                raise ValueError('File has extension \'pickle\' or \'pkl\', but pickle deserialization failed')
            return data_deserialized
        if file_extension.lower() == '.json':
            try:
                data_deserialized = json.load(open(data_object, 'r'))
            except:
                raise ValueError('File has extension \'json\', but JSON deserialization failed')
            return data_deserialized
        raise ValueError('Data object appears to be a filename, but extension \'{}\' isn\'t currently handled'.format(
            file_extension
        ))
    if isinstance(data_object, str):
            try:
                data_deserialized = json.loads(data_object)
            except:
                raise ValueError('Data object is a string but it doesn\'t appear to be a valid filename or valid JSON')
            return data_deserialized
    raise ValueError('Failed to parse data object')

def convert_to_array(data_object):
    if isinstance(data_object, str):
        try:
            data_object = json.loads(data_object)
        except:
            raise ValueError('Array object \'{}\' appears to be string but JSON deserialization fails'.format(
                data_object
            ))
    return np.asarray(data_object)

def convert_to_list(data_object):
    if data_object is None:
        return None
    try:
        if pd.isnull(data_object):
            return None
    except:
        pass
    try:
        if pd.isnull(data_object).all():
            return None
    except:
        pass
    if isinstance(data_object, str):
        try:
            data_object = json.loads(data_object)
        except:
            raise ValueError('List object \'{}\' appears to be string but JSON deserialization fails'.format(
                data_object
            ))
    return list(data_object)

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

def convert_to_lookup_dict(data_object):
    if isinstance(data_object, dict):
        return data_object
    if isinstance(data_object, pd.DataFrame):
        if data_object.index.name is None:
            data_object = data_object.set_index(data_object.columns[0])
        if len(data_object.columns) < 1:
            raise ValueError('Data object appears to tabular, but with fewer than two columns')
        elif len(data_object.columns) == 1:
            return data_object.iloc[:, 0].to_dict()
        else:
            return data_object.to_dict(orient='index')
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

def nearly_equal(df_1, df_2):
    if not df_1.index.equals(df_2.index):
        logger.warning('Indexes of dataframes are not equal')
        return False, pd.DataFrame()
    if not df_1.columns.equals(df_2.columns):
        logger.warning('Column names of dataframes are not equal')
        return False, pd.DataFrame()
    equality_dict = dict()
    for idx in df_1.index:
        equality_dict[idx] = dict()
        for column in df_1.columns:
            equality_dict[idx][column] = False
            try:
                if pd.isnull(df_1.loc[idx, column]) and pd.isnull(df_2.loc[idx, column]):
                    equality_dict[idx][column] = True
            except:
                pass
            try:
                if df_1.loc[idx, column] == df_2.loc[idx, column]:
                    equality_dict[idx][column] = True
            except:
                pass
            try:
                if np.allclose(df_1.loc[idx, column], df_2.loc[idx, column]):
                    equality_dict[idx][column] = True
            except:
                pass
    equality_df = pd.DataFrame.from_dict(equality_dict, orient='index')
    is_equal = np.all(equality_df)
    return is_equal, equality_df

def convert_to_datetime_utc(datetime_object):
    return pd.to_datetime(datetime_object, utc=True).to_pydatetime()
