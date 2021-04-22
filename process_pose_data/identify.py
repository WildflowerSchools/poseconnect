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
    active_person_ids=None,
    ignore_z=False,
    return_match_statistics=False
):
    pose_identification_timestamp_df_list = list()
    if return_match_statistics:
        match_statistics_list = list()
    for timestamp, poses_3d_with_tracks_and_sensor_positions_timestamp_df in poses_3d_with_tracks_and_sensor_positions_df.groupby('timestamp'):
        uwb_data_resampled_timestamp_df = uwb_data_resampled_df.loc[uwb_data_resampled_df['timestamp'] == timestamp]
        if return_match_statistics:
            pose_identification_timestamp_df, match_statistics = identify_poses_timestamp(
                poses_3d_with_tracks_and_sensor_positions_timestamp_df=poses_3d_with_tracks_and_sensor_positions_timestamp_df,
                uwb_data_resampled_timestamp_df=uwb_data_resampled_timestamp_df,
                active_person_ids=active_person_ids,
                ignore_z=ignore_z,
                return_match_statistics=return_match_statistics
            )
            match_statistics_list.append([timestamp] + match_statistics)
        else:
            pose_identification_timestamp_df = identify_poses_timestamp(
                poses_3d_with_tracks_and_sensor_positions_timestamp_df=poses_3d_with_tracks_and_sensor_positions_timestamp_df,
                uwb_data_resampled_timestamp_df=uwb_data_resampled_timestamp_df,
                active_person_ids=active_person_ids,
                ignore_z=ignore_z,
                return_match_statistics=return_match_statistics
            )
        pose_identification_timestamp_df_list.append(pose_identification_timestamp_df)
    pose_identification_df = pd.concat(pose_identification_timestamp_df_list)
    if return_match_statistics:
        match_statistics_df = pd.DataFrame(
            match_statistics_list,
            columns=[
                'timestamp',
                'num_poses',
                'num_persons',
                'num_matches'
            ]
        )
        return pose_identification_df, match_statistics_df
    return pose_identification_df

def identify_poses_timestamp(
    poses_3d_with_tracks_and_sensor_positions_timestamp_df,
    uwb_data_resampled_timestamp_df,
    active_person_ids=None,
    ignore_z=False,
    return_match_statistics=False
):
    num_poses = len(poses_3d_with_tracks_and_sensor_positions_timestamp_df)
    if len(uwb_data_resampled_timestamp_df) > 0:
        if active_person_ids is not None:
            uwb_data_resampled_timestamp_df = uwb_data_resampled_timestamp_df.loc[
                uwb_data_resampled_timestamp_df['person_id'].isin(active_person_ids)
            ]
    num_persons = len(uwb_data_resampled_timestamp_df)
    num_matches = 0
    if num_poses > 0:
        timestamps = poses_3d_with_tracks_and_sensor_positions_timestamp_df['timestamp'].unique()
        if len(timestamps) > 1:
            raise ValueError('3D pose data contains duplicate timestamps')
        timestamp_poses_3d = timestamps[0]
    if num_persons > 0:
        timestamps = uwb_data_resampled_timestamp_df['timestamp'].unique()
        if len(timestamps) > 1:
            raise ValueError('UWB data contains duplicate timestamps')
        timestamp_uwb_data = timestamps[0]
    if num_poses == 0 and num_persons == 0:
        logger.warn('No 3D pose data or UWB data for this (unknown) timestamp')
    if num_poses == 0 and num_persons != 0:
        logger.warn('No 3D pose data for timestamp %s', timestamp_uwb_data.isoformat())
    if num_poses != 0 and num_persons == 0:
        logger.warn('No UWB data for timestamp %s', timestamp_poses_3d.isoformat())
    if num_poses == 0 or num_persons == 0:
        if return_match_statistics:
            match_statistics = [num_poses, num_persons, num_matches]
            return pd.DataFrame(), match_statistics
        return pd.DataFrame()
    if num_poses != 0 and num_persons != 0 and timestamp_poses_3d != timestamp_uwb_data:
        raise ValueError('Timestamp in 3D pose data is {} but timestamp in UWB data is {}'.format(
            timestamp_poses_3d.isoformat(),
            timestamp_uwb_data.isoformat()
        ))
    timestamp = timestamp_poses_3d
    pose_track_3d_ids = poses_3d_with_tracks_and_sensor_positions_timestamp_df['pose_track_3d_id'].values
    person_ids = uwb_data_resampled_timestamp_df['person_id'].values
    if ignore_z:
        pose_track_3d_positions = poses_3d_with_tracks_and_sensor_positions_timestamp_df.loc[:, ['x_position', 'y_position']].values
        person_positions = uwb_data_resampled_timestamp_df.loc[:, ['x_position', 'y_position']].values
    else:
        pose_track_3d_positions = poses_3d_with_tracks_and_sensor_positions_timestamp_df.loc[:, ['x_position', 'y_position', 'z_position']].values
        person_positions = uwb_data_resampled_timestamp_df.loc[:, ['x_position', 'y_position', 'z_position']].values
    distance_matrix = np.zeros((num_poses, num_persons))
    for i in range(num_poses):
        for j in range(num_persons):
            distance_matrix[i, j] = np.linalg.norm(pose_track_3d_positions[i] - person_positions[j])
    pose_track_3d_indices, person_indices = scipy.optimize.linear_sum_assignment(distance_matrix)
    num_expected_matches = min(num_poses, num_persons)
    num_matches = len(pose_track_3d_indices)
    if num_matches != num_expected_matches:
        raise ValueError('Matching {} poses and {} persons so expected {} matches but found {} matches. Distance matrix: {}'.format(
            num_poses,
            num_persons,
            num_expected_matches,
            num_matches,
            distance_matrix
        ))
    pose_identification_timestamp_df = pd.DataFrame({
        'timestamp': timestamp,
        'pose_track_3d_id': pose_track_3d_ids[pose_track_3d_indices],
        'person_id': person_ids[person_indices]
    })
    if return_match_statistics:
        match_statistics = [num_poses, num_persons, num_matches]
        return pose_identification_timestamp_df, match_statistics
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
        max_matches = np.max(person_id_counts)
        total_matches = np.sum(person_id_counts)
        histogram = list(zip(person_ids, person_id_counts))
        pose_track_identification_list.append({
            'pose_track_3d_id': pose_track_3d_id,
            'person_id': person_id,
            'max_matches': max_matches,
            'total_matches': total_matches,
            'histogram': histogram
        })
    pose_track_identification_df = pd.DataFrame(pose_track_identification_list)
    return pose_track_identification_df
