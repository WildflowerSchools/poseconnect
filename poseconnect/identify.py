import poseconnect.utils
import poseconnect.defaults
import pandas as pd
import numpy as np
import scipy
import logging

logger = logging.getLogger(__name__)

def identify_pose_tracks_3d(
    poses_3d_with_tracks,
    sensor_data,
    frames_per_second=poseconnect.defaults.FRAMES_PER_SECOND,
    id_field_names=poseconnect.defaults.IDENTIFICATION_ID_FIELD_NAMES,
    interpolation_field_names=poseconnect.defaults.IDENTIFICATION_INTERPOLATION_FIELD_NAMES,
    timestamp_field_name=poseconnect.defaults.IDENTIFICATION_TIMESTAMP_FIELD_NAME,
    sensor_position_keypoint_index=poseconnect.defaults.IDENTIFICATION_SENSOR_POSITION_KEYPOINT_INDEX,
    active_person_ids=poseconnect.defaults.IDENTIFICATION_ACTIVE_PERSON_IDS,
    ignore_z=poseconnect.defaults.IDENTIFICATION_IGNORE_Z,
    max_distance=poseconnect.defaults.IDENTIFICATION_MAX_DISTANCE,
    min_fraction_matched=poseconnect.defaults.IDENTIFICATION_MIN_TRACK_FRACTION_MATCHED
):
    poses_3d_with_tracks = poseconnect.utils.ingest_poses_3d_with_tracks(poses_3d_with_tracks)
    sensor_data_resampled = resample_sensor_data(
        sensor_data=sensor_data,
        frames_per_second=frames_per_second,
        id_field_names=id_field_names,
        interpolation_field_names=interpolation_field_names,
        timestamp_field_name=timestamp_field_name
    )
    pose_identification = generate_pose_identification(
        poses_3d_with_tracks=poses_3d_with_tracks,
        sensor_data_resampled = sensor_data_resampled,
        sensor_position_keypoint_index=sensor_position_keypoint_index,
        active_person_ids=active_person_ids,
        ignore_z=ignore_z,
        max_distance=max_distance,
        return_match_statistics=False
    )
    pose_track_identification = generate_pose_track_identification(
        pose_identification = pose_identification
    )
    num_poses = poses_3d_with_tracks.groupby('pose_track_3d_id').size().to_frame(name='num_poses')
    pose_track_identification = pose_track_identification.join(num_poses, on='pose_track_3d_id')
    pose_track_identification['fraction_matched'] = pose_track_identification['max_matches']/pose_track_identification['num_poses']
    if min_fraction_matched is not None:
        pose_track_identification = pose_track_identification.loc[pose_track_identification['fraction_matched'] >= min_fraction_matched]
    poses_3d_with_person_ids = (
        poses_3d_with_tracks
        .join(
            pose_track_identification.set_index('pose_track_3d_id')['person_id'],
            how='left',
            on='pose_track_3d_id'
        )
    )
    return poses_3d_with_person_ids

def resample_sensor_data(
    sensor_data,
    frames_per_second=poseconnect.defaults.FRAMES_PER_SECOND,
    id_field_names=poseconnect.defaults.IDENTIFICATION_ID_FIELD_NAMES,
    interpolation_field_names=poseconnect.defaults.IDENTIFICATION_INTERPOLATION_FIELD_NAMES,
    timestamp_field_name=poseconnect.defaults.IDENTIFICATION_TIMESTAMP_FIELD_NAME
):
    sensor_data = poseconnect.utils.ingest_sensor_data(
        data_object=sensor_data,
        id_field_names=id_field_names
    )
    if sensor_data.duplicated().any():
        logger.warning('Duplicate position records found in sensor data. Deleting duplicates.')
        sensor_data.drop_duplicates(inplace=True)
    if len(sensor_data) == 0:
        return sensor_data
    sensor_data_resampled = (
        sensor_data
        .reset_index()
        .set_index(timestamp_field_name)
        .groupby(id_field_names)
        .apply(
            lambda group: resample_sensor_data_person(
                sensor_data_person=group,
                frames_per_second=frames_per_second,
                interpolation_field_names=interpolation_field_names
            )
        )
        .reset_index()
        .reindex(columns = [timestamp_field_name] + id_field_names + interpolation_field_names)
    )
    return sensor_data_resampled

def resample_sensor_data_person(
    sensor_data_person,
    frames_per_second=poseconnect.defaults.FRAMES_PER_SECOND,
    interpolation_field_names=poseconnect.defaults.IDENTIFICATION_INTERPOLATION_FIELD_NAMES
):
    if not isinstance(frames_per_second, int):
        raise ValueError('Only integer frame rates currently supported')
    if not 1000 % frames_per_second == 0:
        raise ValueError('Only frame periods with integer number of milliseconds currently supported')
    frame_period_milliseconds = 1000//frames_per_second
    frame_period_string = '{}ms'.format(frame_period_milliseconds)
    sensor_data_person = sensor_data_person.reindex(columns=interpolation_field_names)
    old_index = sensor_data_person.index
    new_index = pd.date_range(
        start = old_index.min().ceil(frame_period_string),
        end = old_index.max().floor(frame_period_string),
        freq = frame_period_string,
        name='timestamp'
    )
    combined_index = old_index.union(new_index).sort_values()
    sensor_data_person = sensor_data_person.reindex(combined_index)
    sensor_data_person = sensor_data_person.interpolate(method='time')
    sensor_data_person = sensor_data_person.reindex(new_index)
    return sensor_data_person

def generate_pose_identification(
    poses_3d_with_tracks,
    sensor_data_resampled,
    sensor_position_keypoint_index=poseconnect.defaults.IDENTIFICATION_SENSOR_POSITION_KEYPOINT_INDEX,
    active_person_ids=poseconnect.defaults.IDENTIFICATION_ACTIVE_PERSON_IDS,
    ignore_z=poseconnect.defaults.IDENTIFICATION_IGNORE_Z,
    max_distance=poseconnect.defaults.IDENTIFICATION_MAX_DISTANCE,
    return_match_statistics=poseconnect.defaults.IDENTIFICATION_RETURN_MATCH_STATISTICS
):
    sensor_position_keypoint_index = poseconnect.utils.ingest_sensor_position_keypoint_index(sensor_position_keypoint_index)
    pose_identification_timestamp_list = list()
    if return_match_statistics:
        match_statistics_list = list()
    for timestamp, poses_3d_with_tracks_timestamp in poses_3d_with_tracks.groupby('timestamp'):
        sensor_data_resampled_timestamp = sensor_data_resampled.loc[sensor_data_resampled['timestamp'] == timestamp]
        if return_match_statistics:
            pose_identification_timestamp, match_statistics = generate_pose_identification_timestamp(
                poses_3d_with_tracks_timestamp=poses_3d_with_tracks_timestamp,
                sensor_data_resampled_timestamp=sensor_data_resampled_timestamp,
                sensor_position_keypoint_index=sensor_position_keypoint_index,
                active_person_ids=active_person_ids,
                ignore_z=ignore_z,
                return_match_statistics=return_match_statistics
            )
            match_statistics_list.append([timestamp] + match_statistics)
        else:
            pose_identification_timestamp = generate_pose_identification_timestamp(
                poses_3d_with_tracks_timestamp=poses_3d_with_tracks_timestamp,
                sensor_data_resampled_timestamp=sensor_data_resampled_timestamp,
                sensor_position_keypoint_index=sensor_position_keypoint_index,
                active_person_ids=active_person_ids,
                ignore_z=ignore_z,
                max_distance=max_distance,
                return_match_statistics=return_match_statistics
            )
        pose_identification_timestamp_list.append(pose_identification_timestamp)
    pose_identification = pd.concat(pose_identification_timestamp_list)
    if return_match_statistics:
        match_statistics = pd.DataFrame(
            match_statistics_list,
            columns=[
                'timestamp',
                'num_poses',
                'num_persons',
                'num_matches'
            ]
        )
        return pose_identification, match_statistics
    return pose_identification

def generate_pose_identification_timestamp(
    poses_3d_with_tracks_timestamp,
    sensor_data_resampled_timestamp,
    sensor_position_keypoint_index=poseconnect.defaults.IDENTIFICATION_SENSOR_POSITION_KEYPOINT_INDEX,
    active_person_ids=poseconnect.defaults.IDENTIFICATION_ACTIVE_PERSON_IDS,
    ignore_z=poseconnect.defaults.IDENTIFICATION_IGNORE_Z,
    max_distance=poseconnect.defaults.IDENTIFICATION_MAX_DISTANCE,
    return_match_statistics=poseconnect.defaults.IDENTIFICATION_RETURN_MATCH_STATISTICS
):
    num_poses = len(poses_3d_with_tracks_timestamp)
    if len(sensor_data_resampled_timestamp) > 0:
        if active_person_ids is not None:
            sensor_data_resampled_timestamp = sensor_data_resampled_timestamp.loc[
                sensor_data_resampled_timestamp['person_id'].isin(active_person_ids)
            ].copy()
    num_persons = len(sensor_data_resampled_timestamp)
    num_matches = 0
    if num_poses > 0:
        timestamps = poses_3d_with_tracks_timestamp['timestamp'].unique()
        if len(timestamps) > 1:
            raise ValueError('3D pose data contains duplicate timestamps')
        timestamp_poses_3d = timestamps[0]
    if num_persons > 0:
        timestamps = sensor_data_resampled_timestamp['timestamp'].unique()
        if len(timestamps) > 1:
            raise ValueError('UWB data contains duplicate timestamps')
        timestamp_sensor_data = timestamps[0]
    if num_poses == 0 and num_persons == 0:
        logger.warn('No 3D pose data or UWB data for this (unknown) timestamp')
    if num_poses == 0 and num_persons != 0:
        logger.warn('No 3D pose data for timestamp %s', timestamp_sensor_data.isoformat())
    if num_poses != 0 and num_persons == 0:
        logger.warn('No UWB data for timestamp %s', timestamp_poses_3d.isoformat())
    if num_poses == 0 or num_persons == 0:
        if return_match_statistics:
            match_statistics = [num_poses, num_persons, num_matches]
            return pd.DataFrame(), match_statistics
        return pd.DataFrame()
    if num_poses != 0 and num_persons != 0 and timestamp_poses_3d != timestamp_sensor_data:
        raise ValueError('Timestamp in 3D pose data is {} but timestamp in UWB data is {}'.format(
            timestamp_poses_3d.isoformat(),
            timestamp_sensor_data.isoformat()
        ))
    timestamp = timestamp_poses_3d
    pose_track_3d_ids = poses_3d_with_tracks_timestamp['pose_track_3d_id'].values
    person_ids = sensor_data_resampled_timestamp['person_id'].values
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
            keypoints = poses_3d_with_tracks_timestamp.iloc[i]['keypoint_coordinates_3d']
            if keypoint_index is not None and np.all(np.isfinite(keypoints[keypoint_index])):
                pose_track_position = keypoints[keypoint_index]
            else:
                pose_track_position = np.nanmedian(keypoints, axis=0)
            person_position = sensor_data_resampled_timestamp.iloc[j][['x_position', 'y_position', 'z_position']].values
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
    pose_identification_timestamp = pd.DataFrame({
        'timestamp': timestamp,
        'pose_track_3d_id': pose_track_3d_ids[pose_track_3d_indices],
        'person_id': person_ids[person_indices]
    })
    if return_match_statistics:
        match_statistics = [num_poses, num_persons, num_matches]
        return pose_identification_timestamp, match_statistics
    return pose_identification_timestamp


def generate_pose_track_identification(
    pose_identification
):
    pose_track_identification_list = list()
    for pose_track_3d_id, pose_identification_pose_track in pose_identification.groupby('pose_track_3d_id'):
        person_ids, person_id_counts = np.unique(
            pose_identification_pose_track['person_id'],
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
    pose_track_identification = pd.DataFrame(pose_track_identification_list)
    return pose_track_identification
