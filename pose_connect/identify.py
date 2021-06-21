import pandas as pd
import numpy as np
import scipy
import logging

logger = logging.getLogger(__name__)

def generate_track_identification(
    poses_3d_with_tracks_df,
    uwb_data_df,
    sensor_position_keypoint_index=None
):
    uwb_data_resampled_df = resample_uwb_data(uwb_data_df)
    identification_df = calculate_track_identification(
        poses_3d_with_tracks_df=poses_3d_with_tracks_df,
        uwb_data_resampled_df=uwb_data_resampled_df,
        sensor_position_keypoint_index=sensor_position_keypoint_index
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

def calculate_track_identification(
    poses_3d_with_tracks_df,
    uwb_data_resampled_df,
    sensor_position_keypoint_index=None
):
    pose_identification_df = identify_poses(
        poses_3d_with_tracks_df=poses_3d_with_tracks_df,
        uwb_data_resampled_df=uwb_data_resampled_df,
        sensor_position_keypoint_index=sensor_position_keypoint_index
    )
    pose_track_identification_df = identify_pose_tracks(
        pose_identification_df=pose_identification_df
    )
    return pose_track_identification_df

def identify_poses(
    poses_3d_with_tracks_df,
    uwb_data_resampled_df,
    sensor_position_keypoint_index=None,
    active_person_ids=None,
    ignore_z=False,
    max_distance=None,
    return_match_statistics=False
):
    pose_identification_timestamp_df_list = list()
    if return_match_statistics:
        match_statistics_list = list()
    for timestamp, poses_3d_with_tracks_timestamp_df in poses_3d_with_tracks_df.groupby('timestamp'):
        uwb_data_resampled_timestamp_df = uwb_data_resampled_df.loc[uwb_data_resampled_df['timestamp'] == timestamp]
        if return_match_statistics:
            pose_identification_timestamp_df, match_statistics = identify_poses_timestamp(
                poses_3d_with_tracks_timestamp_df=poses_3d_with_tracks_timestamp_df,
                uwb_data_resampled_timestamp_df=uwb_data_resampled_timestamp_df,
                sensor_position_keypoint_index=sensor_position_keypoint_index,
                active_person_ids=active_person_ids,
                ignore_z=ignore_z,
                return_match_statistics=return_match_statistics
            )
            match_statistics_list.append([timestamp] + match_statistics)
        else:
            pose_identification_timestamp_df = identify_poses_timestamp(
                poses_3d_with_tracks_timestamp_df=poses_3d_with_tracks_timestamp_df,
                uwb_data_resampled_timestamp_df=uwb_data_resampled_timestamp_df,
                sensor_position_keypoint_index=sensor_position_keypoint_index,
                active_person_ids=active_person_ids,
                ignore_z=ignore_z,
                max_distance=max_distance,
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
    poses_3d_with_tracks_timestamp_df,
    uwb_data_resampled_timestamp_df,
    sensor_position_keypoint_index=None,
    active_person_ids=None,
    ignore_z=False,
    max_distance=None,
    return_match_statistics=False
):
    num_poses = len(poses_3d_with_tracks_timestamp_df)
    if len(uwb_data_resampled_timestamp_df) > 0:
        if active_person_ids is not None:
            uwb_data_resampled_timestamp_df = uwb_data_resampled_timestamp_df.loc[
                uwb_data_resampled_timestamp_df['person_id'].isin(active_person_ids)
            ].copy()
    num_persons = len(uwb_data_resampled_timestamp_df)
    num_matches = 0
    if num_poses > 0:
        timestamps = poses_3d_with_tracks_timestamp_df['timestamp'].unique()
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
    pose_track_3d_ids = poses_3d_with_tracks_timestamp_df['pose_track_3d_id'].values
    person_ids = uwb_data_resampled_timestamp_df['person_id'].values
    distance_matrix = np.zeros((num_poses, num_persons))
    for i in range(num_poses):
        for j in range(num_persons):
            if sensor_position_keypoint_index is None:
                keypoint_index = None
            elif isinstance(sensor_position_keypoint_index, int):
                keypoint_index = sensor_position_keypoint_index
            elif isinstance(sensor_position_keypoint_index, dict):
                keypoint_index = sensor_position_keypoint_index.get(person_ids[j])
            else:
                raise ValueError('Sensor position keypoint index specification must be int or dict or None')
            keypoints = poses_3d_with_tracks_timestamp_df.iloc[i]['keypoint_coordinates_3d']
            if keypoint_index is not None and np.all(np.isfinite(keypoints[keypoint_index])):
                pose_track_position = keypoints[keypoint_index]
            else:
                pose_track_position = np.nanmedian(keypoints, axis=0)
            person_position = uwb_data_resampled_timestamp_df.iloc[j][['x_position', 'y_position', 'z_position']].values
            displacement_vector = pose_track_position - person_position
            if ignore_z:
                displacement_vector = displacement_vector[:2]
            distance_matrix[i, j] = np.linalg.norm(displacement_vector)
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
    if max_distance is not None:
        new_pose_track_3d_indices=list()
        new_person_indices=list()
        for pose_track_3d_index, person_index in zip(pose_track_3d_indices, person_indices):
            if distance_matrix[pose_track_3d_index, person_index] <= max_distance:
                new_pose_track_3d_indices.append(pose_track_3d_index)
                new_person_indices.append(person_index)
        pose_track_3d_indices=np.asarray(new_pose_track_3d_indices)
        person_indices=np.asarray(new_person_indices)
        num_matches = len(pose_track_3d_indices)
    if num_matches == 0:
        if return_match_statistics:
            match_statistics = [num_poses, num_persons, num_matches]
            return pd.DataFrame(), match_statistics
        else:
            return pd.DataFrame()
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
