import pandas as pd
import numpy as np
import scipy
import logging

logger = logging.getLogger(__name__)

def generate_track_identification(
    poses_3d_with_tracks_df,
    uwb_data_df,
    sensor_position_keypoint_index=10
):
    uwb_data_resampled_df = resample_uwb_data(uwb_data_df)
    poses_3d_with_tracks_and_sensor_positions_df = extract_sensor_position_data(
        poses_3d_with_tracks_df=poses_3d_with_tracks_df,
        sensor_position_keypoint_index=sensor_position_keypoint_index
    )
    identification_df = calculate_track_identification(
        poses_3d_with_tracks_and_sensor_positions_df=poses_3d_with_tracks_and_sensor_positions_df,
        uwb_data_resampled_df=uwb_data_resampled_df
    )
    return identification_df

# def resample_uwb_data(
#     uwb_data_df
# ):
#     uwb_data_resampled_df = (
#         uwb_data_df
#         .set_index('timestamp')
#         .groupby('person_id')
#         .apply(resample_uwb_data_person)
#         .reset_index()
#     )
#     return uwb_data_resampled_df
#
# def resample_uwb_data_person(
#     uwb_data_person_df
# ):
#     uwb_data_person_df = uwb_data_person_df.drop(columns='person_id')
#     old_index = uwb_data_person_df.index
#     new_index = pd.date_range(
#         start = old_index.min().ceil('100ms'),
#         end = old_index.max().floor('100ms'),
#         freq = '100ms',
#         name='timestamp'
#     )
#     combined_index = old_index.union(new_index).sort_values()
#     uwb_data_person_combined_index_df = uwb_data_person_df.reindex(combined_index)
#     uwb_data_person_combined_index_interpolated_df = uwb_data_person_combined_index_df.interpolate(method='time')
#     uwb_data_person_new_index_interpolated_df = uwb_data_person_combined_index_interpolated_df.reindex(new_index)
#     return uwb_data_person_new_index_interpolated_df

def resample_uwb_data(
    uwb_data_df,
    id_field_names=['person_id'],
    interpolation_field_names = ['x_position', 'y_position', 'z_position'],
    timestamp_field_name='timestamp'
):
    if len(uwb_data_df) == 0:
        return uwb_data_df
    uwb_data_resampled_df = (
        uwb_data_df
        .reset_index()
        .set_index(timestamp_field_name)
        .groupby(id_field_names)
        .apply(
            lambda group_df: resample_uwb_data_person(
                uwb_data_person_df=group_df,
                interpolation_field_names=interpolation_field_names
            )
        )
        .reset_index()
        .reindex(columns = [timestamp_field_name] + id_field_names + interpolation_field_names)
    )
    return uwb_data_resampled_df

def resample_uwb_data_person(
    uwb_data_person_df,
    interpolation_field_names = ['x_position', 'y_position', 'z_position']
):
    uwb_data_person_df = uwb_data_person_df.reindex(columns=interpolation_field_names)
    old_index = uwb_data_person_df.index
    new_index = pd.date_range(
        start = old_index.min().ceil('100ms'),
        end = old_index.max().floor('100ms'),
        freq = '100ms',
        name='timestamp'
    )
    combined_index = old_index.union(new_index).sort_values()
    uwb_data_person_df = uwb_data_person_df.reindex(combined_index)
    uwb_data_person_df = uwb_data_person_df.interpolate(method='time')
    uwb_data_person_df = uwb_data_person_df.reindex(new_index)
    return uwb_data_person_df

def extract_sensor_position_data(
    poses_3d_with_tracks_df,
    sensor_position_keypoint_index=10
):
    poses_3d_with_tracks_and_sensor_positions_df = poses_3d_with_tracks_df.copy()
    poses_3d_with_tracks_and_sensor_positions_df['x_position'] = (
        poses_3d_with_tracks_and_sensor_positions_df['keypoint_coordinates_3d']
        .apply(lambda x: x[sensor_position_keypoint_index, 0])
    )
    poses_3d_with_tracks_and_sensor_positions_df['y_position'] = (
        poses_3d_with_tracks_and_sensor_positions_df['keypoint_coordinates_3d']
        .apply(lambda x: x[sensor_position_keypoint_index, 1])
    )
    poses_3d_with_tracks_and_sensor_positions_df['z_position'] = (
        poses_3d_with_tracks_and_sensor_positions_df['keypoint_coordinates_3d']
        .apply(lambda x: x[sensor_position_keypoint_index, 2])
    )
    return poses_3d_with_tracks_and_sensor_positions_df

def calculate_track_identification(
    poses_3d_with_tracks_and_sensor_positions_df,
    uwb_data_resampled_df
):
    pose_identification_df = identify_poses(
        poses_3d_with_tracks_and_sensor_positions_df=poses_3d_with_tracks_and_sensor_positions_df,
        uwb_data_resampled_df=uwb_data_resampled_df
    )
    pose_track_identification_df = identify_pose_tracks(
        pose_identification_df=pose_identification_df
    )
    return pose_track_identification_df

def identify_poses(
    poses_3d_with_tracks_and_sensor_positions_df,
    uwb_data_resampled_df,
    active_person_ids=None
):
    pose_identification_timestamp_df_list = list()
    for timestamp, poses_3d_with_tracks_and_sensor_positions_timestamp_df in poses_3d_with_tracks_and_sensor_positions_df.groupby('timestamp'):
        uwb_data_resampled_timestamp_df = uwb_data_resampled_df.loc[uwb_data_resampled_df['timestamp'] == timestamp]
        pose_identification_timestamp_df = identify_poses_timestamp(
            poses_3d_with_tracks_and_sensor_positions_timestamp_df=poses_3d_with_tracks_and_sensor_positions_timestamp_df,
            uwb_data_resampled_timestamp_df=uwb_data_resampled_timestamp_df,
            active_person_ids=active_person_ids
        )
        pose_identification_timestamp_df_list.append(pose_identification_timestamp_df)
    pose_identification_df = pd.concat(pose_identification_timestamp_df_list)
    return pose_identification_df

def identify_poses_timestamp(
    poses_3d_with_tracks_and_sensor_positions_timestamp_df,
    uwb_data_resampled_timestamp_df,
    active_person_ids=None
):
    timestamp_poses_3d = None
    if len(poses_3d_with_tracks_and_sensor_positions_timestamp_df) > 0:
        timestamps = poses_3d_with_tracks_and_sensor_positions_timestamp_df['timestamp'].unique()
        if len(timestamps) > 1:
            raise ValueError('3D pose data contains duplicate timestamps')
        timestamp_poses_3d = timestamps[0]
    timestamp_uwb_data = None
    if len(uwb_data_resampled_timestamp_df) > 0:
        timestamps = uwb_data_resampled_timestamp_df['timestamp'].unique()
        if len(timestamps) > 1:
            raise ValueError('UWB data contains duplicate timestamps')
        timestamp_uwb_data = timestamps[0]
    if timestamp_poses_3d is None and timestamp_uwb_data is None:
        logger.warn('No 3D pose data or UWB data for this (unknown) timestamp')
    if timestamp_poses_3d is None and timestamp_uwb_data is not None:
        logger.warn('No 3D pose data for timestamp %s', timestamp_uwb_data.isoformat())
        return pd.DataFrame()
    if timestamp_uwb_data is None and timestamp_poses_3d is not None:
        logger.warn('No UWB data for timestamp %s', timestamp_poses_3d.isoformat())
        return pd.DataFrame()
    if timestamp_poses_3d is not None and timestamp_uwb_data is not None and timestamp_poses_3d != timestamp_uwb_data:
        raise ValueError('Timestamp in 3D pose data is {} but timestamp in UWB data is {}'.format(
            timestamp_poses_3d.isoformat(),
            timestamp_uwb_data.isoformat()
        ))
    timestamp = timestamp_poses_3d
    num_pose_tracks_3d = len(poses_3d_with_tracks_and_sensor_positions_timestamp_df)
    pose_track_3d_ids = poses_3d_with_tracks_and_sensor_positions_timestamp_df['pose_track_3d_id'].values
    pose_track_3d_positions = poses_3d_with_tracks_and_sensor_positions_timestamp_df.loc[:, ['x_position', 'y_position', 'z_position']].values
    if active_person_ids is not None:
        uwb_data_resampled_timestamp_df = uwb_data_resampled_timestamp_df.loc[uwb_data_resampled_timestamp_df['person_id'].isin(active_person_ids)]
    num_persons = len(uwb_data_resampled_timestamp_df)
    person_ids = uwb_data_resampled_timestamp_df['person_id'].values
    uwb_positions = uwb_data_resampled_timestamp_df.loc[:, ['x_position', 'y_position', 'z_position']].values
    distance_matrix = np.zeros((num_pose_tracks_3d, num_persons))
    for i in range(num_pose_tracks_3d):
        for j in range(num_persons):
            distance_matrix[i, j] = np.linalg.norm(pose_track_3d_positions[i] - uwb_positions[j])
    pose_track_3d_indices, person_indices = scipy.optimize.linear_sum_assignment(distance_matrix)
    pose_identification_timestamp_df = pd.DataFrame({
        'timestamp': timestamp,
        'pose_track_3d_id': pose_track_3d_ids[pose_track_3d_indices],
        'person_id': person_ids[person_indices]
    })
    return pose_identification_timestamp_df

def identify_pose_tracks(
    pose_identification_df
):
    pose_track_identification_list = list()
    for pose_track_3d_id, pose_identification_pose_track_df in pose_identification_df.groupby('pose_track_3d_id'):
        person_ids, person_id_counts = np.unique(
            pose_identification_pose_track_df['person_id'],
            return_counts=True
        )
        person_id = person_ids[np.argmax(person_id_counts)]
        histogram = list(zip(person_ids, person_id_counts))
        pose_track_identification_list.append({
            'pose_track_3d_id': pose_track_3d_id,
            'person_id': person_id,
            'histogram': histogram
        })
    pose_track_identification_df = pd.DataFrame(pose_track_identification_list)
    return pose_track_identification_df
