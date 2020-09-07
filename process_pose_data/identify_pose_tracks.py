import pandas as pd
import numpy as np
import scipy


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
    uwb_data_df
):
    uwb_data_resampled_df = (
        uwb_data_df
        .set_index('timestamp')
        .groupby('person_id')
        .apply(resample_uwb_data_person)
        .reset_index()
    )
    return uwb_data_resampled_df

def resample_uwb_data_person(
    uwb_data_person_df
):
    uwb_data_person_df = uwb_data_person_df.drop(columns='person_id')
    old_index = uwb_data_person_df.index
    new_index = pd.date_range(
        start = old_index.min().ceil('100ms'),
        end = old_index.max().floor('100ms'),
        freq = '100ms',
        name='timestamp'
    )
    combined_index = old_index.union(new_index).sort_values()
    uwb_data_person_combined_index_df = uwb_data_person_df.reindex(combined_index)
    uwb_data_person_combined_index_interpolated_df = uwb_data_person_combined_index_df.interpolate(method='time')
    uwb_data_person_new_index_interpolated_df = uwb_data_person_combined_index_interpolated_df.reindex(new_index)
    return uwb_data_person_new_index_interpolated_df

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
    indentication_df_by_timestamp_list = list()
    for timestamp, poses_3d_with_tracks_df_timestamp in poses_3d_with_tracks_and_sensor_positions_df.groupby('timestamp'):
        uwb_data_df_timestamp = uwb_data_resampled_df.loc[uwb_data_resampled_df['timestamp'] == timestamp]
        if len(uwb_data_df_timestamp) == 0:
            logging.warn('No UWB data for timestamp %s', timestamp.isoformat())
            continue
        num_pose_tracks = len(poses_3d_with_tracks_df_timestamp)
        pose_track_ids = poses_3d_with_tracks_df_timestamp['pose_track_3d_id'].values
        pose_track_positions = poses_3d_with_tracks_df_timestamp.loc[:, ['x_position', 'y_position', 'z_position']].values
        num_persons = len(uwb_data_df_timestamp)
        person_ids = uwb_data_df_timestamp['person_id'].values
        uwb_positions = uwb_data_df_timestamp.loc[:, ['x_position', 'y_position', 'z_position']].values
        distance_matrix = np.zeros((num_pose_tracks, num_persons))
        for i in range(num_pose_tracks):
            for j in range(num_persons):
                distance_matrix[i, j] = np.linalg.norm(pose_track_positions[i] - uwb_positions[j])
        pose_track_indices, person_indices = scipy.optimize.linear_sum_assignment(distance_matrix)
        identification_df_timestamp = pd.DataFrame({
            'timestamp': timestamp,
            'pose_track_3d_id': pose_track_ids[pose_track_indices],
            'person_id': person_ids[person_indices]
        })
        indentication_df_by_timestamp_list.append(identification_df_timestamp)
    identification_df_by_timestamp = pd.concat(indentication_df_by_timestamp_list)
    identification_data_list = list()
    for pose_track_id, identification_df_pose_track_id in identification_df_by_timestamp.groupby('pose_track_3d_id'):
        person_ids, person_id_counts = np.unique(
            identification_df_pose_track_id['person_id'],
            return_counts=True
        )
        person_id = person_ids[np.argmax(person_id_counts)]
        histogram = list(zip(person_ids, person_id_counts))
        identification_data_list.append({
            'pose_track_3d_id': pose_track_id,
            'person_id': person_id,
            'histogram': histogram
        })
    identification_df = pd.DataFrame(identification_data_list)
    return identification_df
