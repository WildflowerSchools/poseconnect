import honeycomb_io
import pandas as pd
import numpy as np
import logging
from uuid import uuid4
import datetime
import dateutil
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
    tree_structure='file-per-frame',
    filename='alphapose-results.json',
    json_format='cmu'
):
    time_segment_start_utc = time_segment_start.astimezone(datetime.timezone.utc)
    glob_pattern = alphapose_data_file_glob_pattern(
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
        tree_structure=tree_structure,
        filename=filename
    )
    re_pattern = alphapose_data_file_re_pattern(
        base_dir=base_dir,
        alphapose_subdirectory=alphapose_subdirectory,
        tree_structure=tree_structure,
        filename=filename
    )
    data_list = list()
    if tree_structure == 'file-per-frame':
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
            frame_number = int(m.group('frame_number_string'))
            with open(path, 'r') as fp:
                pose_data_object = json.load(fp)
            if len(pose_data_object) == 0:
                continue
            timestamp_json_string = pose_data_object.get('timestamp')
            assignment_id_json = pose_data_object.get('assignment_id')
            environment_id_json = pose_data_object.get('environment_id')
            try:
                timestamp_json = dateutil.parser.isoparse(timestamp_json_string)
            except:
                raise ValueError('Timestamp string in JSON \'{}\' cannot be parsed by dateutil.parser.isoparse()'.format(
                    timestamp_json_string
                ))
            timestamp_path = timestamp_video_file + datetime.timedelta(microseconds = 10**5*frame_number)
            if timestamp_json != timestamp_path:
                raise ValueError('Timestamp in JSON \'{}\' does not match timestamp inferred from path \'{}\' for file \'{}\' (extracted frame number string \'{}\' resulting in frame number {})'.format(
                    timestamp_json.isoformat(),
                    timestamp_path.isoformat(),
                    path,
                    m.group('frame_number_string'),
                    frame_number
                ))
            if assignment_id_json != assignment_id:
                raise ValueError('Assignment ID in JSON \'{}\' does not match assignment ID inferred from path \'{}\' for file \'{}\''.format(
                    assignment_id_json,
                    assignment_id,
                    path
                ))
            if environment_id_json != environment_id:
                raise ValueError('Assignment ID in JSON \'{}\' does not match assignment ID inferred from path \'{}\' for file \'{}\''.format(
                    environment_id_json,
                    environment_id,
                    path
                ))
            poses = pose_data_object.get('poses')
            if poses is None:
                raise ValueError('JSON in file \'{}\' does not contain \'poses\' field')
            if len(poses) == 0:
                continue
            for pose in poses:
                keypoints = np.asarray([[keypoint.get('x'), keypoint.get('y')] for keypoint in pose.get('keypoints')])
                keypoint_quality = np.asarray([keypoint.get('quality')for keypoint in pose.get('keypoints')])
                keypoints = np.where(keypoints == 0.0, np.nan, keypoints)
                keypoint_quality = np.where(keypoint_quality == 0.0, np.nan, keypoint_quality)
                pose_quality = pose.get('quality')
                pose_2d_id = pose.get('pose_id')
                data_list.append({
                    'pose_2d_id': pose_2d_id,
                    'timestamp': pd.to_datetime(timestamp_json),
                    'assignment_id': assignment_id_json,
                    'keypoint_coordinates_2d': keypoints,
                    'keypoint_quality_2d': keypoint_quality,
                    'pose_quality_2d': pose_quality
                })
    elif tree_structure == 'file-per-segment':
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
                            'pose_2d_id': uuid4().hex,
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
                        'pose_2d_id': uuid4().hex,
                        'timestamp': pd.to_datetime(timestamp),
                        'assignment_id': assignment_id,
                        'keypoint_coordinates_2d': keypoints,
                        'keypoint_quality_2d': keypoint_quality,
                        'pose_quality_2d': pose_quality
                    })
            else:
                raise ValueError('JSON format specifier \'{}\' not recognized'.format(json_format))
    else:
        raise ValueError('Tree structure specification \'{}\' not recognized'.format(
            tree_structure
        ))
    df = pd.DataFrame(data_list)
    if len(df) == 0:
        logger.warning('No poses found for time segment starting at %04d/%02d/%02dT%02d:%02d:%02d. Returning empty data frame',
            time_segment_start_utc.year,
            time_segment_start_utc.month,
            time_segment_start_utc.day,
            time_segment_start_utc.hour,
            time_segment_start_utc.minute,
            time_segment_start_utc.second
        )
        return df
    df.set_index('pose_2d_id', inplace=True)
    df.sort_values(['timestamp', 'assignment_id'], inplace=True)
    return df

def fetch_3d_poses_with_person_info(
    base_dir,
    environment_id,
    pose_track_3d_identification_inference_id,
    start=None,
    end=None,
    pose_processing_subdirectory='pose_processing',
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    poses_3d_with_tracks_identified_df = fetch_3d_poses_with_identified_tracks_local(
        base_dir=base_dir,
        environment_id=environment_id,
        pose_track_3d_identification_inference_id=pose_track_3d_identification_inference_id,
        start=start,
        end=end,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    person_info_df = honeycomb_io.fetch_person_info(
        environment_id=environment_id,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    poses_3d_with_person_info_df = poses_3d_with_tracks_identified_df.join(
        person_info_df,
        on='person_id'
    )
    return poses_3d_with_person_info_df

def fetch_3d_poses_with_identified_tracks_local(
    base_dir,
    environment_id,
    pose_track_3d_identification_inference_id,
    start=None,
    end=None,
    pose_processing_subdirectory='pose_processing'
):
    pose_track_3d_identification_metadata = fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_track_3d_identification',
        environment_id=environment_id,
        filename_stem='pose_track_3d_identification_metadata',
        inference_ids=pose_track_3d_identification_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=None,
        object_type='dict',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    if start is None:
        start = pose_track_3d_identification_metadata['parameters']['start']
    if end is None:
        end = pose_track_3d_identification_metadata['parameters']['end']
    pose_track_3d_interpolation_inference_id = pose_track_3d_identification_metadata['parameters']['pose_track_3d_interpolation_inference_id']
    poses_3d_with_tracks_df = fetch_3d_poses_with_interpolated_tracks_local(
        base_dir=base_dir,
        environment_id=environment_id,
        pose_track_3d_interpolation_inference_id=pose_track_3d_interpolation_inference_id,
        start=start,
        end=end,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    pose_track_identification_df = fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_track_3d_identification',
        environment_id=environment_id,
        filename_stem='pose_track_3d_identification',
        inference_ids=pose_track_3d_identification_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=None,
        object_type='dataframe',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    poses_3d_with_tracks_identified_df = (
        poses_3d_with_tracks_df
        .join(
            pose_track_identification_df
            .set_index('pose_track_3d_id'),
            on='pose_track_3d_id'
        )
    )
    return poses_3d_with_tracks_identified_df

def fetch_3d_poses_with_interpolated_tracks_local(
    base_dir,
    environment_id,
    pose_track_3d_interpolation_inference_id,
    start=None,
    end=None,
    pose_processing_subdirectory='pose_processing'
):
    pose_track_3d_interpolation_metadata = fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_track_3d_interpolation',
        environment_id=environment_id,
        filename_stem='pose_track_3d_interpolation_metadata',
        inference_ids=pose_track_3d_interpolation_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=None,
        object_type='dict',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    if start is None:
        start = pose_track_3d_interpolation_metadata['parameters']['start']
    if end is None:
        end = pose_track_3d_interpolation_metadata['parameters']['end']
    pose_tracking_3d_inference_id = pose_track_3d_interpolation_metadata['parameters']['pose_tracking_3d_inference_id']
    pose_reconstruction_3d_inference_id = pose_track_3d_interpolation_metadata['parameters']['pose_reconstruction_3d_inference_id']
    poses_3d_with_tracks_before_interpolation_df = fetch_3d_poses_with_tracks_local(
        base_dir=base_dir,
        environment_id=environment_id,
        pose_reconstruction_3d_inference_id=pose_reconstruction_3d_inference_id,
        pose_tracking_3d_inference_id=pose_tracking_3d_inference_id,
        start=start,
        end=end,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    poses_3d_with_tracks_from_interpolation_df = fetch_3d_poses_with_tracks_local(
        base_dir=base_dir,
        environment_id=environment_id,
        pose_reconstruction_3d_inference_id=pose_track_3d_interpolation_inference_id,
        pose_tracking_3d_inference_id=pose_track_3d_interpolation_inference_id,
        start=start,
        end=end,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    poses_3d_with_tracks_df = pd.concat((
        poses_3d_with_tracks_before_interpolation_df,
        poses_3d_with_tracks_from_interpolation_df
    )).sort_values(['pose_track_3d_id', 'timestamp'])
    return poses_3d_with_tracks_df

def fetch_3d_poses_with_uninterpolated_tracks_local(
    base_dir,
    environment_id,
    pose_tracking_3d_inference_id,
    start=None,
    end=None,
    pose_processing_subdirectory='pose_processing'
):
    pose_tracks_3d_metadata = fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_tracking_3d',
        environment_id=environment_id,
        filename_stem='pose_tracking_3d_metadata',
        inference_ids=pose_tracking_3d_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=None,
        object_type='dict',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    if start is None:
        start = pose_tracks_3d_metadata['parameters']['start']
    if end is None:
        end = pose_tracks_3d_metadata['parameters']['end']
    pose_reconstruction_3d_inference_id = pose_tracks_3d_metadata['parameters']['pose_reconstruction_3d_inference_id']
    poses_3d_with_tracks_df = fetch_3d_poses_with_tracks_local(
        base_dir=base_dir,
        environment_id=environment_id,
        pose_reconstruction_3d_inference_id=pose_reconstruction_3d_inference_id,
        pose_tracking_3d_inference_id=pose_tracking_3d_inference_id,
        start=start,
        end=end,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    return poses_3d_with_tracks_df

def fetch_3d_poses_with_tracks_local(
    base_dir,
    environment_id,
    start,
    end,
    pose_reconstruction_3d_inference_id,
    pose_tracking_3d_inference_id,
    pose_processing_subdirectory='pose_processing'
):
    poses_3d_df = fetch_data_local_by_time_segment(
        start=start,
        end=end,
        base_dir=base_dir,
        pipeline_stage='pose_reconstruction_3d',
        environment_id=environment_id,
        filename_stem='poses_3d',
        inference_ids=pose_reconstruction_3d_inference_id,
        data_ids=None,
        sort_field=None,
        object_type='dataframe',
        pose_processing_subdirectory='pose_processing'
    )
    pose_tracks_3d = fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_tracking_3d',
        environment_id=environment_id,
        filename_stem='pose_tracks_3d',
        inference_ids=pose_tracking_3d_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=None,
        object_type='dict',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    pose_tracks_3d_df = convert_pose_tracks_3d_to_df(pose_tracks_3d)
    poses_3d_with_tracks_df = poses_3d_df.join(
        pose_tracks_3d_df,
        how='inner'
    )
    return poses_3d_with_tracks_df

def write_data_local_by_time_segment(
    data_object,
    base_dir,
    pipeline_stage,
    environment_id,
    filename_stem,
    inference_id,
    object_type='dataframe',
    append=False,
    sort_field=None,
    pose_processing_subdirectory='pose_processing'
):
    if object_type != 'dataframe':
        raise ValueError('Writing data by time segment only available for dataframe objects')
    if 'timestamp' not in data_object.columns.tolist():
        raise ValueError('Writing data by time segment only available for dataframes with a \'timestamp\' field')
    start = pd.to_datetime(data_object['timestamp'].min()).to_pydatetime()
    end = pd.to_datetime(data_object['timestamp'].max()).to_pydatetime()
    time_segment_start_list = generate_time_segment_start_list(
        start,
        end
    )
    for time_segment_start in time_segment_start_list:
        data_object_time_segment = data_object.loc[
            (data_object['timestamp'] >= time_segment_start) &
            (data_object['timestamp'] < time_segment_start + datetime.timedelta(seconds=10))
        ]
        write_data_local(
            data_object=data_object_time_segment,
            base_dir=base_dir,
            pipeline_stage=pipeline_stage,
            environment_id=environment_id,
            filename_stem=filename_stem,
            inference_id=inference_id,
            time_segment_start=time_segment_start,
            object_type=object_type,
            append=append,
            sort_field=sort_field,
            pose_processing_subdirectory=pose_processing_subdirectory
        )


def write_data_local(
    data_object,
    base_dir,
    pipeline_stage,
    environment_id,
    filename_stem,
    inference_id,
    time_segment_start=None,
    object_type='dataframe',
    append=False,
    sort_field=None,
    pose_processing_subdirectory='pose_processing'
):
    directory_path, filename = data_file_path(
        base_dir=base_dir,
        pipeline_stage=pipeline_stage,
        environment_id=environment_id,
        filename_stem=filename_stem,
        inference_id=inference_id,
        time_segment_start=time_segment_start,
        object_type=object_type,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(
        directory_path,
        filename
    )
    if append and os.path.exists(file_path):
        if object_type != 'dataframe':
            raise ValueError('Append and sort field options only available for dataframe objects')
        existing_data_object = fetch_data_local(
            base_dir=base_dir,
            pipeline_stage=pipeline_stage,
            environment_id=environment_id,
            filename_stem=filename_stem,
            inference_ids=inference_id,
            time_segment_start=time_segment_start,
            object_type=object_type,
            pose_processing_subdirectory=pose_processing_subdirectory
        )
        data_object = pd.concat((existing_data_object, data_object))
        if sort_field is not None:
            data_object.sort_values(sort_field, inplace=True)
    if object_type == 'dataframe':
        data_object.to_pickle(file_path)
    elif object_type == 'dict':
        with open(file_path, 'wb') as fp:
            pickle.dump(data_object, fp)
    else:
        raise ValueError('Only allowed object types are \'dataframe\' and \'dict\'')

def fetch_data_local_by_time_segment(
    start,
    end,
    base_dir,
    pipeline_stage,
    environment_id,
    filename_stem,
    inference_ids,
    data_ids=None,
    sort_field=None,
    object_type='dataframe',
    pose_processing_subdirectory='pose_processing'
):
    if object_type != 'dataframe':
        raise ValueError('Fetching data by time segment only available for dataframe objects')
    time_segment_start_list = generate_time_segment_start_list(
        start,
        end
    )
    data_object_list = list()
    for time_segment_start in time_segment_start_list:
        data_object_time_segment = fetch_data_local(
            base_dir=base_dir,
            pipeline_stage=pipeline_stage,
            environment_id=environment_id,
            filename_stem=filename_stem,
            inference_ids=inference_ids,
            data_ids=data_ids,
            sort_field=sort_field,
            time_segment_start=time_segment_start,
            object_type=object_type,
            pose_processing_subdirectory=pose_processing_subdirectory
        )
        data_object_list.append(data_object_time_segment)
    data_object = pd.concat(data_object_list)
    if sort_field is not None:
        data_object.sort_values(sort_field, inplace=True)
    return data_object

def fetch_data_local(
    base_dir,
    pipeline_stage,
    environment_id,
    filename_stem,
    inference_ids,
    data_ids=None,
    sort_field=None,
    time_segment_start=None,
    object_type='dataframe',
    pose_processing_subdirectory='pose_processing'
):
    if isinstance(inference_ids, str):
        inference_ids = [inference_ids]
    elif isinstance(inference_ids, (list, tuple, set)):
        pass
    else:
        raise ValueError('Specified inference IDs must be of type str, list, tuple, or set')
    if len(inference_ids) == 0:
        raise ValueError('Must specify at least one inference ID')
    data_object_list = list()
    for inference_id in inference_ids:
        directory_path, filename = data_file_path(
            base_dir=base_dir,
            pipeline_stage=pipeline_stage,
            environment_id=environment_id,
            filename_stem=filename_stem,
            inference_id=inference_id,
            time_segment_start=time_segment_start,
            object_type=object_type,
            pose_processing_subdirectory=pose_processing_subdirectory
        )
        file_path = os.path.join(
            directory_path,
            filename
        )
        if object_type == 'dataframe':
            if os.path.exists(file_path):
                data_object_item = pd.read_pickle(file_path)
                if data_ids is not None:
                    data_object_item = data_object_item.reindex(
                        data_object_item.index.intersection(data_ids)
                    )
            else:
                data_object_item = pd.DataFrame()
        elif object_type == 'dict':
            if os.path.exists(file_path):
                with open(file_path, 'rb') as fp:
                    data_object_item = pickle.load(fp)
                if data_ids is not None:
                    raise ValueError('Specification of data IDs is only available for dataframe objects')
            else:
                data_object_item = dict()
        else:
            raise ValueError('Only allowed object types are \'dataframe\' and \'dict\'')
        data_object_list.append(data_object_item)
    if len(data_object_list) == 1:
        data_object = data_object_list[0]
        return data_object
    else:
        if object_type != 'dataframe':
            raise ValueError('Specification of multiple inference IDs is only available for dataframe objects')
        data_object = pd.concat(data_object_list)
        if sort_field is not None:
            data_object.sort_values(sort_field, inplace=True)
    return data_object

def delete_data_local(
    base_dir,
    pipeline_stage,
    environment_id,
    filename_stem,
    inference_ids,
    time_segment_start=None,
    object_type='dataframe',
    pose_processing_subdirectory='pose_processing'
):
    if isinstance(inference_ids, str):
        inference_ids = [inference_ids]
    elif isinstance(inference_ids, (list, tuple, set)):
        pass
    else:
        raise ValueError('Specified inference IDs must be of type str, list, tuple, or set')
    for inference_id in inference_ids:
        directory_path, filename = data_file_path(
            base_dir=base_dir,
            pipeline_stage=pipeline_stage,
            environment_id=environment_id,
            filename_stem=filename_stem,
            inference_id=inference_id,
            time_segment_start=time_segment_start,
            object_type=object_type,
            pose_processing_subdirectory=pose_processing_subdirectory
        )
        file_path = os.path.join(
            directory_path,
            filename
        )
        if os.path.exists(file_path):
            os.remove(file_path)

def data_file_path(
    base_dir,
    pipeline_stage,
    environment_id,
    filename_stem,
    inference_id,
    time_segment_start=None,
    object_type='dataframe',
    pose_processing_subdirectory='pose_processing'
):
    directory_path = os.path.join(
        base_dir,
        pose_processing_subdirectory,
        pipeline_stage,
        environment_id
    )
    if time_segment_start is not None:
        time_segment_start_utc = time_segment_start.astimezone(datetime.timezone.utc)
        directory_path = os.path.join(
            directory_path,
            '{:04d}'.format(time_segment_start_utc.year),
            '{:02d}'.format(time_segment_start_utc.month),
            '{:02d}'.format(time_segment_start_utc.day),
            '{:02d}-{:02d}-{:02d}'.format(
                time_segment_start_utc.hour,
                time_segment_start_utc.minute,
                time_segment_start_utc.second,
            )
        )
    filename = '{}_{}.pkl'.format(
        filename_stem,
        inference_id
    )
    return directory_path, filename

def convert_pose_tracks_3d_to_df(
    pose_tracks_3d
):
    pose_3d_ids_with_tracks_df_list = list()
    for pose_track_3d_id, pose_track_3d in pose_tracks_3d.items():
        pose_3d_ids_with_tracks_single_track_df = pd.DataFrame(
            {'pose_track_3d_id': pose_track_3d_id},
            index=pose_track_3d['pose_3d_ids']
        )
        pose_3d_ids_with_tracks_single_track_df.index.name='pose_3d_id'
        pose_3d_ids_with_tracks_df_list.append(pose_3d_ids_with_tracks_single_track_df)
    pose_3d_ids_with_tracks_df = pd.concat(pose_3d_ids_with_tracks_df_list)
    return pose_3d_ids_with_tracks_df

def add_short_track_labels(
    poses_3d_with_tracks_df,
    pose_track_3d_id_column_name='pose_track_3d_id'
):
    pose_track_3d_id_index = poses_3d_with_tracks_df.groupby(pose_track_3d_id_column_name).apply(lambda x: x['timestamp'].min()).sort_values().index
    track_label_lookup = pd.DataFrame(
        range(1, len(pose_track_3d_id_index)+1),
        columns=['pose_track_3d_id_short'],
        index=pose_track_3d_id_index
    )
    poses_3d_with_tracks_df = poses_3d_with_tracks_df.join(track_label_lookup, on='pose_track_3d_id')
    return poses_3d_with_tracks_df

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
    frame_number=None,
    alphapose_subdirectory='prepared',
    tree_structure='file-per-frame',
    filename='alphapose-results.json'
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
    if frame_number is not None:
        frame_number_string = '{:d}'.format(frame_number)
    else:
        frame_number_string = '*'
    if tree_structure == 'file-per-frame':
        glob_pattern = os.path.join(
            base_dir_string,
            alphapose_subdirectory_string,
            environment_id_string,
            camera_assignment_id_string,
            year_string,
            month_string,
            day_string,
            '-'.join([hour_string, minute_string, second_string]),
            'poses-{}.json'.format(frame_number_string)
        )
    elif tree_structure == 'file-per-segment':
        glob_pattern = os.path.join(
            base_dir_string,
            alphapose_subdirectory_string,
            environment_id_string,
            camera_assignment_id_string,
            year_string,
            month_string,
            day_string,
            '-'.join([hour_string, minute_string, second_string]),
            filename
        )
    else:
        raise ValueError('Tree structure specification \'{}\' not recognized'.format(
            tree_structure
        ))
    return glob_pattern

def alphapose_data_file_re_pattern(
    base_dir,
    alphapose_subdirectory='prepared',
    tree_structure='file-per-frame',
    filename='alphapose-results.json'
):
    if tree_structure=='file-per-frame':
        re_pattern = os.path.join(
            base_dir,
            alphapose_subdirectory,
            '(?P<environment_id>.+)',
            '(?P<assignment_id>.+)',
            '(?P<year_string>[0-9]{4})',
            '(?P<month_string>[0-9]{2})',
            '(?P<day_string>[0-9]{2})',
            '(?P<hour_string>[0-9]{2})\-(?P<minute_string>[0-9]{2})\-(?P<second_string>[0-9]{2})',
            'poses-(?P<frame_number_string>[0-9]+)\.json'
        )
    elif tree_structure=='file-per-segment':
        re_pattern = os.path.join(
            base_dir,
            alphapose_subdirectory,
            '(?P<environment_id>.+)',
            '(?P<assignment_id>.+)',
            '(?P<year_string>[0-9]{4})',
            '(?P<month_string>[0-9]{2})',
            '(?P<day_string>[0-9]{2})',
            '(?P<hour_string>[0-9]{2})\-(?P<minute_string>[0-9]{2})\-(?P<second_string>[0-9]{2})',
            filename
        )
    return re_pattern

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
        camera_device_id_lookup = honeycomb_io.fetch_camera_device_id_lookup(
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
