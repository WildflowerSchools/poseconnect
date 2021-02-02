import process_pose_data.honeycomb_io
import pandas as pd
import numpy as np
import logging
from uuid import uuid4
import datetime
import os
import glob
import pickle
import re
import json
import math

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
                if isinstance(obj, datetime.datetime):
                        return obj.isoformat()
                if isinstance(obj, np.ndarray):
                        return obj.tolist()
                return json.JSONEncoder.default(self, obj)

def fetch_2d_pose_data_alphapose_local_time_segment(
    base_dir,
    environment_id,
    time_segment_start,
    alphapose_subdirectory='prepared',
    file_name='alphapose-results.json',
    json_format='cmu'
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
        alphapose_subdirectory=alphapose_subdirectory,
        file_name=file_name,
        json_format=json_format
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
    alphapose_subdirectory='prepared',
    file_name='alphapose-results.json',
    json_format='cmu'
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
        alphapose_subdirectory=alphapose_subdirectory,
        file_name=file_name
    )
    re_pattern = alphapose_data_file_re_pattern(
        base_dir=base_dir,
        alphapose_subdirectory=alphapose_subdirectory,
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
            pose_data_object = json.load(fp)
        if len(pose_data_object) == 0:
            continue
        if json_format == 'cmu':
            # JSON is a dict structure with an entry for each image
            for image_filename, pose_data in pose_data_object.items():
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
                        'pose_2d_id_local': uuid4().hex,
                        'timestamp': pd.to_datetime(timestamp),
                        'assignment_id': assignment_id,
                        'keypoint_coordinates_2d': keypoints,
                        'keypoint_quality_2d': keypoint_quality,
                        'pose_quality_2d': pose_quality
                    })
        elif json_format == 'list':
            # JSON is a list structure with an item for each pose
            for pose_data in pose_data_object:
                frame_number = int(pose_data.get('image_id').split('.')[0])
                timestamp = timestamp_video_file + datetime.timedelta(microseconds = 10**5*frame_number)
                keypoint_data_array = np.asarray(pose_data['keypoints']).reshape((-1 , 3))
                keypoints = keypoint_data_array[:, :2]
                keypoint_quality = keypoint_data_array[:, 2]
                keypoints = np.where(keypoints == 0.0, np.nan, keypoints)
                keypoint_quality = np.where(keypoint_quality == 0.0, np.nan, keypoint_quality)
                pose_quality = pose_data.get('score')
                data_list.append({
                    'pose_2d_id_local': uuid4().hex,
                    'timestamp': pd.to_datetime(timestamp),
                    'assignment_id': assignment_id,
                    'keypoint_coordinates_2d': keypoints,
                    'keypoint_quality_2d': keypoint_quality,
                    'pose_quality_2d': pose_quality
                })
        else:
            raise ValueError('JSON format specifier \'{}\' not recognized'.format(json_format))
    df = pd.DataFrame(data_list)
    if len(df) == 0:
        logger.warning('No poses found for time segment starting at %04d/%02d/%02dT%02d:%02d:%02d. Returning empty data frame',
            year,
            month,
            day,
            hour,
            minute,
            second
        )
        return df
    df.set_index('pose_2d_id_local', inplace=True)
    df.sort_values(['timestamp', 'assignment_id'], inplace=True)
    return df

def write_3d_pose_data_local(
    poses_3d_df,
    base_dir,
    environment_id,
    inference_id_local,
    append=False,
    pose_processing_subdirectory='pose_processing',
    poses_3d_directory_name='poses_3d',
    poses_3d_file_name_stem='poses_3d'
):
    start = pd.to_datetime(poses_3d_df['timestamp'].min()).to_pydatetime()
    end = pd.to_datetime(poses_3d_df['timestamp'].max()).to_pydatetime()
    time_segment_start_list = generate_time_segment_start_list(
        start,
        end
    )
    for time_segment_start in time_segment_start_list:
        poses_3d_time_segment_df = poses_3d_df.loc[
            (poses_3d_df['timestamp'] >= time_segment_start) &
            (poses_3d_df['timestamp'] < time_segment_start + datetime.timedelta(seconds=10))
        ]
        write_3d_pose_data_local_time_segment(
            poses_3d_df=poses_3d_time_segment_df,
            base_dir=base_dir,
            environment_id=environment_id,
            time_segment_start=time_segment_start,
            inference_id_local=inference_id_local,
            append=append,
            pose_processing_subdirectory=pose_processing_subdirectory,
            poses_3d_directory_name=poses_3d_directory_name,
            poses_3d_file_name_stem=poses_3d_file_name_stem
        )

def write_3d_pose_data_local_time_segment(
    poses_3d_df,
    base_dir,
    environment_id,
    time_segment_start,
    inference_id_local,
    append=False,
    pose_processing_subdirectory='pose_processing',
    poses_3d_directory_name='poses_3d',
    poses_3d_file_name_stem='poses_3d'
):
    directory_path = pose_3d_data_directory_path_time_segment(
        base_dir=base_dir,
        environment_id=environment_id,
        time_segment_start=time_segment_start,
        pose_processing_subdirectory=pose_processing_subdirectory,
        poses_3d_directory_name=poses_3d_directory_name
    )
    file_name = '{}_{}.pkl'.format(
        poses_3d_file_name_stem,
        inference_id_local
    )
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(
        directory_path,
        file_name
    )
    if append and os.path.exists(file_path):
        existing_poses_3d_df = pd.read_pickle(file_path)
        poses_3d_df = pd.concat((existing_poses_3d_df, poses_3d_df)).sort_values('timestamp')
    poses_3d_df.to_pickle(file_path)

def fetch_3d_pose_data_local(
    start,
    end,
    base_dir,
    environment_id,
    inference_id_local,
    pose_3d_ids=None,
    pose_processing_subdirectory='pose_processing',
    poses_3d_directory_name='poses_3d',
    poses_3d_file_name_stem='poses_3d'
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
            inference_id_local=inference_id_local,
            pose_3d_ids=pose_3d_ids,
            pose_processing_subdirectory=pose_processing_subdirectory,
            poses_3d_directory_name=poses_3d_directory_name,
            poses_3d_file_name_stem=poses_3d_file_name_stem
        )
        poses_3d_df_list.append(poses_3d_df_time_segment)
    poses_3d_df = pd.concat(poses_3d_df_list)
    return poses_3d_df

def fetch_3d_pose_data_local_time_segment(
    time_segment_start,
    base_dir,
    environment_id,
    inference_id_local,
    pose_3d_ids=None,
    pose_processing_subdirectory='pose_processing',
    poses_3d_directory_name='poses_3d',
    poses_3d_file_name_stem='poses_3d'
):
    if isinstance(inference_id_local, str):
        path=pose_3d_data_path_time_segment(
            base_dir=base_dir,
            environment_id=environment_id,
            inference_id_local=inference_id_local,
            time_segment_start=time_segment_start,
            pose_processing_subdirectory=pose_processing_subdirectory,
            poses_3d_directory_name=poses_3d_directory_name,
            poses_3d_file_name_stem=poses_3d_file_name_stem
        )
        if os.path.exists(path):
            poses_3d_df_time_segment = pd.read_pickle(path)
        else:
            poses_3d_df_time_segment = pd.DataFrame()
    elif isinstance(inference_id_local, (list, tuple, set)):
        poses_3d_dfs_time_segment=list()
        for inference_id_local_element in inference_id_local:
            path=pose_3d_data_path_time_segment(
                base_dir=base_dir,
                environment_id=environment_id,
                inference_id_local=inference_id_local_element,
                time_segment_start=time_segment_start,
                pose_processing_subdirectory=pose_processing_subdirectory,
                poses_3d_directory_name=poses_3d_directory_name,
                poses_3d_file_name_stem=poses_3d_file_name_stem
            )
            if os.path.exists(path):
                poses_3d_df_time_segment_id = pd.read_pickle(path)
            else:
                poses_3d_df_time_segment_id = pd.DataFrame()
            poses_3d_dfs_time_segment.append(poses_3d_df_time_segment_id)
        poses_3d_df_time_segment = pd.concat(poses_3d_dfs_time_segment).sort_values('timestamp')
    else:
        raise ValueError("Specified inference ID must be of type str, list, tuple, or set")
    if pose_3d_ids is not None:
        poses_3d_df_time_segment = poses_3d_df_time_segment.reindex(
            poses_3d_df_time_segment.index.intersection(pose_3d_ids)
        )
    return poses_3d_df_time_segment

def delete_3d_pose_data_local(
    base_dir,
    environment_id,
    inference_id_local,
    pose_processing_subdirectory='pose_processing',
    poses_3d_directory_name='poses_3d',
    poses_3d_file_name_stem='poses_3d'
):
    glob_pattern = os.path.join(
        base_dir,
        pose_processing_subdirectory,
        environment_id,
        poses_3d_directory_name,
        '*',
        '*',
        '*',
        '*',
        '{}_{}.pkl'.format(poses_3d_file_name_stem, inference_id_local)
    )
    for path in glob.iglob(glob_pattern):
        os.remove(path)

def write_3d_pose_track_data_local(
    pose_tracks_3d,
    base_dir,
    environment_id,
    inference_id_local,
    pose_processing_subdirectory='pose_processing',
    pose_tracks_3d_directory_name='pose_tracks_3d',
    pose_tracks_3d_file_name_stem='pose_tracks_3d'
):
    directory = os.path.join(
        base_dir,
        pose_processing_subdirectory,
        environment_id,
        pose_tracks_3d_directory_name
    )
    filename = '{}_{}.pkl'.format(
        pose_tracks_3d_file_name_stem,
        inference_id_local
    )
    path = os.path.join(
        directory,
        filename
    )
    os.makedirs(directory, exist_ok=True)
    with open(path, 'wb') as fp:
        pickle.dump(pose_tracks_3d.output(), fp)

def fetch_3d_pose_track_data_local(
    base_dir,
    environment_id,
    inference_id_local,
    pose_processing_subdirectory='pose_processing',
    pose_tracks_3d_directory_name='pose_tracks_3d',
    pose_tracks_3d_file_name_stem='pose_tracks_3d'
):
    path=os.path.join(
        base_dir,
        pose_processing_subdirectory,
        environment_id,
        pose_tracks_3d_directory_name,
        '{}_{}.pkl'.format(
            pose_tracks_3d_file_name_stem,
            inference_id_local
        )
    )
    with open(path, 'rb') as fp:
        pose_tracks_3d=pickle.load(fp)
    return pose_tracks_3d

def delete_3d_pose_track_data_local(
    base_dir,
    environment_id,
    inference_id_local,
    pose_processing_subdirectory='pose_processing',
    pose_tracks_3d_directory_name='pose_tracks_3d',
    pose_tracks_3d_file_name_stem='pose_tracks_3d'
):
    path=os.path.join(
        base_dir,
        pose_processing_subdirectory,
        environment_id,
        pose_tracks_3d_directory_name,
        '{}_{}.pkl'.format(
            pose_tracks_3d_file_name_stem,
            inference_id_local
        )
    )
    if os.path.exists(path):
        os.remove(path)

def write_pose_reconstruction_3d_metadata_local(
    pose_reconstruction_3d_metadata,
    base_dir,
    environment_id,
    pose_processing_subdirectory='pose_processing',
    poses_3d_directory_name='poses_3d',
    pose_reconstruction_3d_metadata_filename_stem='pose_reconstruction_3d_metadata'
):
    write_metadata_local(
        metadata=pose_reconstruction_3d_metadata,
        base_dir=base_dir,
        environment_id=environment_id,
        output_subdirectory_name=poses_3d_directory_name,
        metadata_filename_stem=pose_reconstruction_3d_metadata_filename_stem,
        pose_processing_subdirectory=pose_processing_subdirectory
    )

def write_pose_tracking_3d_metadata_local(
    pose_tracking_3d_metadata,
    base_dir,
    environment_id,
    pose_processing_subdirectory='pose_processing',
    pose_tracks_3d_directory_name='pose_tracks_3d',
    pose_tracking_3d_metadata_filename_stem='pose_tracking_3d_metadata'
):
    write_metadata_local(
        metadata=pose_tracking_3d_metadata,
        base_dir=base_dir,
        environment_id=environment_id,
        output_subdirectory_name=pose_tracks_3d_directory_name,
        metadata_filename_stem=pose_tracking_3d_metadata_filename_stem,
        pose_processing_subdirectory=pose_processing_subdirectory
    )

def write_metadata_local(
    metadata,
    base_dir,
    environment_id,
    output_subdirectory_name,
    metadata_filename_stem,
    pose_processing_subdirectory='pose_processing'
):
    inference_id_local = metadata.get('inference_execution').get('inference_id_local')
    metadata_directory = os.path.join(
        base_dir,
        pose_processing_subdirectory,
        environment_id,
        output_subdirectory_name
    )
    metadata_filename = '{}_{}.json'.format(
        metadata_filename_stem,
        inference_id_local
    )
    metadata_path = os.path.join(
        metadata_directory,
        metadata_filename
    )
    os.makedirs(metadata_directory, exist_ok=True)
    with open(metadata_path, 'w') as fp:
        json.dump(metadata, fp, cls=CustomJSONEncoder, indent=2)

def read_pose_reconstruction_3d_metadata_local(
    inference_id_local,
    base_dir,
    environment_id,
    pose_processing_subdirectory='pose_processing',
    poses_3d_directory_name='poses_3d',
    pose_reconstruction_3d_metadata_filename_stem='pose_reconstruction_3d_metadata'
):
    pose_reconstruction_3d_metadata = read_metadata_local(
        inference_id_local=inference_id_local,
        base_dir=base_dir,
        environment_id=environment_id,
        output_subdirectory_name=poses_3d_directory_name,
        metadata_filename_stem=pose_reconstruction_3d_metadata_filename_stem,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    pose_reconstruction_3d_metadata['start'] = datetime.datetime.fromisoformat(
        pose_reconstruction_3d_metadata.get('start')
    )
    pose_reconstruction_3d_metadata['end'] = datetime.datetime.fromisoformat(
        pose_reconstruction_3d_metadata.get('end')
    )
    pose_reconstruction_3d_metadata['inference_execution']['execution_start'] = datetime.datetime.fromisoformat(
        pose_reconstruction_3d_metadata.get('inference_execution').get('execution_start')
    )
    for camera_id in pose_reconstruction_3d_metadata.get('camera_calibrations').keys():
        pose_reconstruction_3d_metadata['camera_calibrations'][camera_id]['camera_matrix']=np.asarray(
            pose_reconstruction_3d_metadata.get('camera_calibrations').get(camera_id).get('camera_matrix')
        )
        pose_reconstruction_3d_metadata['camera_calibrations'][camera_id]['distortion_coefficients']=np.asarray(
            pose_reconstruction_3d_metadata.get('camera_calibrations').get(camera_id).get('distortion_coefficients')
        )
        pose_reconstruction_3d_metadata['camera_calibrations'][camera_id]['rotation_vector']=np.asarray(
            pose_reconstruction_3d_metadata.get('camera_calibrations').get(camera_id).get('rotation_vector')
        )
        pose_reconstruction_3d_metadata['camera_calibrations'][camera_id]['translation_vector']=np.asarray(
            pose_reconstruction_3d_metadata.get('camera_calibrations').get(camera_id).get('translation_vector')
        )
    pose_reconstruction_3d_metadata['pose_3d_limits']=np.asarray(
        pose_reconstruction_3d_metadata.get('pose_3d_limits')
    )
    return pose_reconstruction_3d_metadata

def read_pose_tracking_3d_metadata_local(
    inference_id_local,
    base_dir,
    environment_id,
    pose_processing_subdirectory='pose_processing',
    pose_tracks_3d_directory_name='pose_tracks_3d',
    pose_tracking_3d_metadata_filename_stem='pose_tracking_3d_metadata'
):
    pose_tracking_3d_metadata = read_metadata_local(
        inference_id_local=inference_id_local,
        base_dir=base_dir,
        environment_id=environment_id,
        output_subdirectory_name=pose_tracks_3d_directory_name,
        metadata_filename_stem=pose_tracking_3d_metadata_filename_stem,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    pose_tracking_3d_metadata['start'] = datetime.datetime.fromisoformat(
        pose_tracking_3d_metadata.get('start')
    )
    pose_tracking_3d_metadata['end'] = datetime.datetime.fromisoformat(
        pose_tracking_3d_metadata.get('end')
    )
    pose_tracking_3d_metadata['inference_execution']['execution_start'] = datetime.datetime.fromisoformat(
        pose_tracking_3d_metadata.get('inference_execution').get('execution_start')
    )
    return pose_tracking_3d_metadata

def read_metadata_local(
    inference_id_local,
    base_dir,
    environment_id,
    output_subdirectory_name,
    metadata_filename_stem,
    pose_processing_subdirectory='pose_processing'
):
    metadata_directory = os.path.join(
        base_dir,
        pose_processing_subdirectory,
        environment_id,
        output_subdirectory_name
    )
    metadata_filename = '{}_{}.json'.format(
        metadata_filename_stem,
        inference_id_local
    )
    metadata_path = os.path.join(
        metadata_directory,
        metadata_filename
    )
    with open(metadata_path, 'r') as fp:
        metadata=json.load(fp)
    return metadata

def delete_pose_reconstruction_3d_metadata_local(
    inference_id_local,
    base_dir,
    environment_id,
    pose_processing_subdirectory='pose_processing',
    poses_3d_directory_name='poses_3d',
    pose_reconstruction_3d_metadata_filename_stem='pose_reconstruction_3d_metadata'
):
    delete_metadata_local(
        inference_id_local=inference_id_local,
        base_dir=base_dir,
        environment_id=environment_id,
        output_subdirectory_name=poses_3d_directory_name,
        metadata_filename_stem=pose_reconstruction_3d_metadata_filename_stem,
        pose_processing_subdirectory=pose_processing_subdirectory
    )

def delete_pose_tracking_3d_metadata_local(
    inference_id_local,
    base_dir,
    environment_id,
    pose_processing_subdirectory='pose_processing',
    pose_tracks_3d_directory_name='pose_tracks_3d',
    pose_tracking_3d_metadata_filename_stem='pose_tracking_3d_metadata'
):
    delete_metadata_local(
        inference_id_local=inference_id_local,
        base_dir=base_dir,
        environment_id=environment_id,
        output_subdirectory_name=pose_tracks_3d_directory_name,
        metadata_filename_stem=pose_tracking_3d_metadata_filename_stem,
        pose_processing_subdirectory=pose_processing_subdirectory
    )

def delete_metadata_local(
    inference_id_local,
    base_dir,
    environment_id,
    output_subdirectory_name,
    metadata_filename_stem,
    pose_processing_subdirectory='pose_processing'
):
    metadata_path = os.path.join(
        base_dir,
        pose_processing_subdirectory,
        environment_id,
        output_subdirectory_name,
        '{}_{}.json'.format(
            metadata_filename_stem,
            inference_id_local
        )
    )
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

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
    alphapose_subdirectory='prepared',
    file_name='alphapose-results.json'
):
    base_dir_string = base_dir
    alphapose_subdirectory_string = alphapose_subdirectory
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
        alphapose_subdirectory_string,
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
    alphapose_subdirectory='prepared',
    file_name='alphapose-results.json'
):
    re_pattern = os.path.join(
        base_dir,
        alphapose_subdirectory,
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
    inference_id_local,
    time_segment_start,
    pose_processing_subdirectory='pose_processing',
    poses_3d_directory_name='poses_3d',
    poses_3d_file_name_stem='poses_3d'
):
    directory_path = pose_3d_data_directory_path_time_segment(
        base_dir=base_dir,
        environment_id=environment_id,
        time_segment_start=time_segment_start,
        pose_processing_subdirectory=pose_processing_subdirectory,
        poses_3d_directory_name=poses_3d_directory_name
    )
    file_name='{}_{}.pkl'.format(
        poses_3d_file_name_stem,
        inference_id_local
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
    pose_processing_subdirectory='pose_processing',
    poses_3d_directory_name='poses_3d'
):
    time_segment_start_utc = time_segment_start.astimezone(datetime.timezone.utc)
    path = os.path.join(
        base_dir,
        pose_processing_subdirectory,
        environment_id,
        poses_3d_directory_name,
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
    num_time_segments = math.ceil((end_utc - start_utc_floor).total_seconds()  / 10.0)
    time_segment_start_list = [start_utc_floor + i*datetime.timedelta(seconds=10) for i in range(num_time_segments)]
    return time_segment_start_list
