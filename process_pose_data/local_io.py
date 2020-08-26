import process_pose_data.honeycomb_io
import pandas as pd
import numpy as np
import logging
from uuid import uuid4
import datetime
import os
import glob
import re
import json
import math

logger = logging.getLogger(__name__)

def fetch_2d_pose_data_alphapose_local_time_segment(
    base_dir,
    environment_id,
    time_segment_start,
    file_name='alphapose-results.json'
):
    time_segment_start_utc = time_segment_start.astimezone(datetime.timezone.utc)
    df = fetch_2d_pose_data_alphapose_local(
        base_dir=base_dir,
        environment_id=environment_id,
        camera_assignment_id=None,
        year=time_segment_start_utc.year,
        month=time_segment_start_utc.month,
        day=time_segment_start_utc.day,
        hour=time_segment_start_utc.hour,
        minute=time_segment_start_utc.minute,
        second=time_segment_start_utc.second,
        file_name=file_name
    )
    return df

def fetch_2d_pose_data_alphapose_local(
    base_dir,
    environment_id=None,
    camera_assignment_id=None,
    year=None,
    month=None,
    day=None,
    hour=None,
    minute=None,
    second=None,
    file_name='alphapose-results.json'
):
    glob_pattern = alphapose_data_file_glob_pattern(
        base_dir=base_dir,
        environment_id=environment_id,
        camera_assignment_id=camera_assignment_id,
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
        file_name=file_name
    )
    re_pattern = alphapose_data_file_re_pattern(
        base_dir=base_dir,
        file_name=file_name
    )
    data_list = list()
    for path in glob.iglob(glob_pattern):
        m = re.match(re_pattern, path)
        if not m:
            raise ValueError('Regular expression does not match path: {}'.format(path))
        assignment_id = m.group('assignment_id')
        timestamp_video_file = datetime.datetime(
            int(m.group('year_string')),
            int(m.group('month_string')),
            int(m.group('day_string')),
            int(m.group('hour_string')),
            int(m.group('minute_string')),
            int(m.group('second_string')),
            tzinfo=datetime.timezone.utc
        )
        with open(path, 'r') as fp:
            pose_data_dict = json.load(fp)
        if len(pose_data_dict) == 0:
            continue
        for image_filename, pose_data in pose_data_dict.items():
            frame_number = int(image_filename.split('.')[0])
            timestamp = timestamp_video_file + datetime.timedelta(microseconds = 10**5*frame_number)
            for pose in pose_data['bodies']:
                keypoint_data_array = np.asarray(pose['joints']).reshape((-1 , 3))
                keypoints = keypoint_data_array[:, :2]
                keypoint_quality = keypoint_data_array[:, 2]
                keypoints = np.where(keypoints == 0.0, np.nan, keypoints)
                keypoint_quality = np.where(keypoint_quality == 0.0, np.nan, keypoint_quality)
                pose_quality = pose.get('score')
                data_list.append({
                    'pose_id_2d': uuid4().hex,
                    'timestamp': pd.to_datetime(timestamp),
                    'assignment_id': assignment_id,
                    'keypoint_coordinates_2d': keypoints,
                    'keypoint_quality_2d': keypoint_quality,
                    'pose_quality_2d': pose_quality
                })
    df = pd.DataFrame(data_list)
    df.set_index('pose_id_2d', inplace=True)
    df.sort_values(['timestamp', 'assignment_id'], inplace=True)
    return df

def write_3d_pose_data_local_time_segment(
    poses_3d_df,
    base_dir,
    environment_id,
    time_segment_start,
    directory_name='poses_3d',
    file_name='poses_3d.pkl'
):
    directory_path = pose_3d_data_directory_path_time_segment(
        base_dir=base_dir,
        environment_id=environment_id,
        time_segment_start=time_segment_start,
        directory_name=directory_name
    )
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(
        directory_path,
        file_name
    )
    poses_3d_df.to_pickle(file_path)

def fetch_3d_pose_data_local(
    start,
    end,
    base_dir,
    environment_id,
    directory_name='poses_3d',
    file_name='poses_3d.pkl'
):
    time_segment_start_list = generate_time_segment_start_list(
        start,
        end
    )
    poses_3d_df_list = list()
    for time_segment_start in time_segment_start_list:
        poses_3d_df_time_segment = fetch_3d_pose_data_local_time_segment(
            time_segment_start,
            base_dir=base_dir,
            environment_id=environment_id,
            directory_name=directory_name,
            file_name=file_name
        )
        poses_3d_df_list.append(poses_3d_df_time_segment)
    poses_3d_df = pd.concat(poses_3d_df_list)
    return poses_3d_df

def fetch_3d_pose_data_local_time_segment(
    time_segment_start,
    base_dir,
    environment_id,
    directory_name='poses_3d',
    file_name='poses_3d.pkl'
):
    path=pose_3d_data_path_time_segment(
        base_dir=base_dir,
        environment_id=environment_id,
        time_segment_start=time_segment_start,
        directory_name=directory_name,
        file_name=file_name
    )
    poses_3d_df_time_segment = pd.read_pickle(path)
    return poses_3d_df_time_segment

def alphapose_data_file_glob_pattern(
    base_dir,
    environment_id=None,
    camera_assignment_id=None,
    year=None,
    month=None,
    day=None,
    hour=None,
    minute=None,
    second=None,
    file_name='alphapose-results.json'
):
    base_dir_string = base_dir
    if environment_id is not None:
        environment_id_string = environment_id
    else:
        environment_id_string = '*'
    if camera_assignment_id is not None:
        camera_assignment_id_string = camera_assignment_id
    else:
        camera_assignment_id_string = '*'
    if year is not None:
        year_string = '{:04d}'.format(year)
    else:
        year_string = '????'
    if month is not None:
        month_string = '{:02d}'.format(month)
    else:
        month_string = '??'
    if day is not None:
        day_string = '{:02d}'.format(day)
    else:
        day_string = '??'
    if hour is not None:
        hour_string = '{:02d}'.format(hour)
    else:
        hour_string = '??'
    if minute is not None:
        minute_string = '{:02d}'.format(minute)
    else:
        minute_string = '??'
    if second is not None:
        second_string = '{:02d}'.format(second)
    else:
        second_string = '??'
    glob_pattern = os.path.join(
        base_dir_string,
        environment_id_string,
        camera_assignment_id_string,
        year_string,
        month_string,
        day_string,
        '-'.join([hour_string, minute_string, second_string]),
        file_name
    )
    return glob_pattern

def alphapose_data_file_re_pattern(
    base_dir,
    file_name='alphapose-results.json'
):
    re_pattern = os.path.join(
        base_dir,
        '(?P<environment_id>.+)',
        '(?P<assignment_id>.+)',
        '(?P<year_string>[0-9]{4})',
        '(?P<month_string>[0-9]{2})',
        '(?P<day_string>[0-9]{2})',
        '(?P<hour_string>[0-9]{2})\-(?P<minute_string>[0-9]{2})\-(?P<second_string>[0-9]{2})',
        file_name
    )
    return re_pattern

def pose_3d_data_path_time_segment(
    base_dir,
    environment_id,
    time_segment_start,
    directory_name='poses_3d',
    file_name='poses_3d.pkl'
):
    directory_path = pose_3d_data_directory_path_time_segment(
        base_dir=base_dir,
        environment_id=environment_id,
        time_segment_start=time_segment_start,
        directory_name=directory_name
    )
    path = os.path.join(
        directory_path,
        file_name
    )
    return path

def pose_3d_data_directory_path_time_segment(
    base_dir,
    environment_id,
    time_segment_start,
    directory_name='poses_3d'
):
    time_segment_start_utc = time_segment_start.astimezone(datetime.timezone.utc)
    path = os.path.join(
        base_dir,
        environment_id,
        directory_name,
        '{:04d}'.format(time_segment_start_utc.year),
        '{:02d}'.format(time_segment_start_utc.month),
        '{:02d}'.format(time_segment_start_utc.day),
        '{:02d}-{:02d}-{:02d}'.format(
            time_segment_start_utc.hour,
            time_segment_start_utc.minute,
            time_segment_start_utc.second,
        )
    )
    return path

def convert_assignment_ids_to_camera_device_ids(
    poses_2d_df,
    camera_device_id_lookup=None,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    if camera_device_id_lookup is None:
        assignment_ids = poses_2d_df['assignment_id'].unique().tolist()
        camera_device_id_lookup = process_pose_data.honeycomb_io.fetch_camera_device_id_lookup(
            assignment_ids=assignment_ids,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    poses_2d_df = poses_2d_df.copy()
    poses_2d_df['camera_id'] = poses_2d_df['assignment_id'].apply(lambda assignment_id: camera_device_id_lookup.get(assignment_id))
    poses_2d_df.drop(columns='assignment_id', inplace=True)
    old_column_order = poses_2d_df.columns.tolist()
    new_column_order = [old_column_order[0], old_column_order[-1]] + old_column_order[1:-1]
    poses_2d_df = poses_2d_df.reindex(columns=new_column_order)
    return poses_2d_df

def generate_time_segment_start_list(
    start,
    end
):
    start_utc = start.astimezone(datetime.timezone.utc)
    end_utc = end.astimezone(datetime.timezone.utc)
    start_utc_floor = datetime.datetime(
        year=start_utc.year,
        month=start_utc.month,
        day=start_utc.day,
        hour=start_utc.hour,
        minute=start_utc.minute,
        second=10*(start_utc.second // 10),
        tzinfo=start_utc.tzinfo
    )
    num_time_segments = math.ceil((end_utc - start_utc_floor).seconds / 10.0)
    time_segment_start_list = [start_utc_floor + i*datetime.timedelta(seconds=10) for i in range(num_time_segments)]
    return time_segment_start_list

# def fetch_2d_pose_data_from_local_json(
#     directory_path
# ):
#     data = list()
#     for directory_entry in os.listdir(directory_path):
#         if re.match(r'.*\.json', directory_entry):
#             logger.info('Retrieving pose data from {}'.format(directory_entry))
#             with open(os.path.join(directory_path, directory_entry), 'r') as fh:
#                 data_this_file = json.load(fh)
#             logger.info('Retrieved {} poses from {}'.format(
#                 len(data_this_file),
#                 directory_entry
#             ))
#             data.extend(data_this_file)
#     logger.info('Retrieved {} poses overall. Parsing')
#     parsed_data = list()
#     for datum in data:
#         parsed_data.append({
#             'pose_id_2d': uuid4().hex,
#             'timestamp': datum.get('timestamp'),
#             'camera_id': datum.get('camera'),
#             'track_label_2d': datum.get('track_label'),
#             'pose_model_id': datum.get('pose_model'),
#             'keypoint_coordinates_2d': np.asarray([keypoint.get('coordinates') for keypoint in datum.get('keypoints')]),
#             'keypoint_quality_2d': np.asarray([keypoint.get('quality') for keypoint in datum.get('keypoints')]),
#             'pose_quality_2d': datum.get('quality')
#         })
#     poses_2d_df = pd.DataFrame(parsed_data)
#     poses_2d_df['timestamp'] = pd.to_datetime(poses_2d_df['timestamp'])
#     if poses_2d_df['pose_model_id'].nunique() > 1:
#         raise ValueError('Returned poses are associated with multiple pose models')
#     poses_2d_df.set_index('pose_id_2d', inplace=True)
#     return poses_2d_df
