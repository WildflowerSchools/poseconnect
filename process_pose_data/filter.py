import process_pose_data.fetch
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def filter_keypoints_by_quality(
    df,
    min_keypoint_quality=None,
    max_keypoint_quality=None
):
    df_filtered = df.copy()
    keypoint_coordinate_arrays = df_filtered['keypoint_coordinates'].values
    num_keypoint_coordinate_arrays = len(keypoint_coordinate_arrays)
    keypoint_coordinates = np.concatenate(keypoint_coordinate_arrays, axis = 0)
    keypoint_quality_arrays = df_filtered['keypoint_quality'].values
    num_keypoint_quality_arrays = len(keypoint_quality_arrays)
    keypoint_quality = np.concatenate(keypoint_quality_arrays)
    if num_keypoint_coordinate_arrays != num_keypoint_quality_arrays:
        raise ValueError('Number of keypoint coordinate arrays ({}) does not match number of keypoint quality arrays ({})'.format(
            num_keypoint_coordinate_arrays,
            num_keypoint_quality_arrays
        ))
    num_spatial_dimensions_per_keypoint = keypoint_coordinates.shape[1]
    if min_keypoint_quality is not None:
        mask = np.less(
            keypoint_quality,
            min_keypoint_quality,
            where=~np.isnan(keypoint_quality)
        )
        keypoint_coordinates[mask] = np.array(num_spatial_dimensions_per_keypoint*[np.nan])
        keypoint_quality[mask] = np.nan
    if max_keypoint_quality is not None:
        mask = np.greater(
            keypoint_quality,
            max_keypoint_quality,
            where=~np.isnan(keypoint_quality)
        )
        keypoint_coordinates[mask] = np.array(num_spatial_dimensions_per_keypoint*[np.nan])
        keypoint_quality[mask] = np.nan
    df_filtered['keypoint_coordinates'] = np.split(keypoint_coordinates, num_keypoint_coordinate_arrays)
    df_filtered['keypoint_quality'] = np.split(keypoint_quality, num_keypoint_quality_arrays)
    return df_filtered

def remove_empty_2d_poses(
    df
):
    df_filtered = df.copy()
    non_empty = df['keypoint_coordinates'].apply(
        lambda x: np.any(np.all(np.isfinite(x), axis=1))
    )
    df_filtered = df_filtered.loc[non_empty].copy()
    return df_filtered

def filter_poses_by_num_valid_keypoints(
    df,
    min_num_keypoints=None,
    max_num_keypoints=None
):
    df_filtered = df.copy()
    num_keypoints = df['keypoint_quality'].apply(
        lambda x: np.count_nonzero(~np.isnan(x))
    )
    if min_num_keypoints is not None:
        df_filtered = df_filtered.loc[num_keypoints >= min_num_keypoints]
    if max_num_keypoints is not None:
        df_filtered = df_filtered.loc[num_keypoints <= max_num_keypoints]
    return df_filtered

def filter_poses_by_quality(
    df,
    min_pose_quality=None,
    max_pose_quality=None
):
    df_filtered = df.copy()
    if min_pose_quality is not None:
        df_filtered = df_filtered.loc[df_filtered['pose_quality'] >= min_pose_quality]
    if max_pose_quality is not None:
        df_filtered = df_filtered.loc[df_filtered['pose_quality'] <= max_pose_quality]
    return df_filtered

def filter_pose_pairs_by_score(
    df,
    min_score=None,
    max_score=None
):
    df_filtered = df.copy()
    if min_score is not None:
        df_filtered = df_filtered.loc[df_filtered['score'] >= min_score]
    if max_score is not None:
        df_filtered = df_filtered.loc[df_filtered['score'] <= max_score]
    return df_filtered

def filter_pose_pairs_by_3d_pose_spatial_limits(
    pose_pairs_2d_df,
    pose_3d_range
):
    valid_3d_poses = pose_pairs_2d_df['keypoint_coordinates_3d'].apply(
        lambda x: process_pose_data.analyze.pose_3d_in_range(x, pose_3d_range)
    )
    pose_pairs_2d_df = pose_pairs_2d_df.loc[valid_3d_poses].copy()
    return pose_pairs_2d_df

def filter_pose_pairs_by_best_match(
    pose_pairs_2d_df_timestamp
):
    pose_pairs_2d_df_timestamp.sort_index(inplace=True)
    best_score_indices = list()
    for group_name, group_df in pose_pairs_2d_df_timestamp.groupby(['camera_id_a', 'camera_id_b']):
        best_score_indices.extend(process_pose_data.analyze.extract_best_score_indices_timestamp_camera_pair(group_df))
    pose_pairs_2d_df_timestamp = pose_pairs_2d_df_timestamp.loc[best_score_indices].copy()
    return pose_pairs_2d_df_timestamp

def remove_empty_3d_poses(
    df
):
    df_filtered = df.copy()
    non_empty = df['keypoint_coordinates_3d'].apply(
        lambda x: np.any(np.all(np.isfinite(x), axis=1))
    )
    df_filtered = df_filtered.loc[non_empty].copy()
    return df_filtered

def remove_empty_reprojected_2d_poses(
    df
):
    df_filtered = df.copy()
    non_empty_a = df['keypoint_coordinates_a_reprojected'].apply(
        lambda x: np.any(np.all(np.isfinite(x), axis=1))
    )
    non_empty_b = df['keypoint_coordinates_b_reprojected'].apply(
        lambda x: np.any(np.all(np.isfinite(x), axis=1))
    )
    df_filtered = df_filtered.loc[non_empty_a & non_empty_b].copy()
    return df_filtered

def remove_invalid_pose_pair_scores(
    df
):
    df_filtered = df.copy()
    valid = ~df['score'].isna()
    df_filtered = df_filtered.loc[valid].copy()
    return df_filtered

def select_random_pose(
    df
):
    return df.sample(1).reset_index().iloc[0].to_dict()

def select_random_pose_pair(
    df
):
    return df.sample(1).reset_index().iloc[0].to_dict()

def select_random_match(
    df
):
    return df.loc[df['match']].sample(1).reset_index().iloc[0].to_dict()

def select_random_timestamp_camera_pair(
    df
):
    timestamp = np.random.choice(df['timestamp'].unique())
    df_filtered = process_pose_data.select_pose_pairs(df, timestamp=timestamp)
    camera_id_a = np.random.choice(df_filtered['camera_id_a'].unique())
    df_filtered = process_pose_data.select_pose_pairs(df_filtered, camera_id_a=camera_id_a)
    camera_id_b = np.random.choice(df_filtered['camera_id_b'].unique())
    df_filtered = process_pose_data.select_pose_pairs(df_filtered, camera_id_b=camera_id_b)
    return df_filtered

def select_random_timestamp(
    df
):
    timestamp = np.random.choice(df['timestamp'].unique())
    df_filtered = process_pose_data.select_pose_pairs(df, timestamp=timestamp)
    return df_filtered

def select_random_camera_pair(
    df
):
    camera_id_a = np.random.choice(df['camera_id_a'].unique())
    df_filtered = process_pose_data.select_pose_pairs(df, camera_id_a=camera_id_a)
    camera_id_b = np.random.choice(df_filtered['camera_id_b'].unique())
    df_filtered = process_pose_data.select_pose_pairs(df_filtered, camera_id_b=camera_id_b)
    return df_filtered

def select_pose(
    df,
    pose_id=None,
    timestamp=None,
    camera_id=None,
    camera_name=None,
    track_label=None,
    camera_names=None
):
    df_selected = select_poses(
        df,
        pose_id=pose_id,
        timestamp=timestamp,
        camera_id=camera_id,
        camera_name=camera_name,
        track_label=track_label,
        camera_names=camera_names
    )
    if len(df_selected) == 0:
        raise ValueError('No poses matched criteria')
    if len(df_selected) > 1:
        raise ValueError('Multiple poses matched criteria')
    return(df_selected.reset_index().iloc[0].to_dict())

def select_poses(
    df,
    pose_id=None,
    timestamp=None,
    camera_id=None,
    camera_name=None,
    track_label=None,
    camera_names=None
):
    filter_list = list()
    if pose_id is not None:
        pose_id_filter = (df.index == pose_id)
        filter_list.append(pose_id_filter)
    if timestamp is not None:
        timestamp_pandas = pd.to_datetime(timestamp, utc=True)
        timestamp_filter = (df['timestamp'] == timestamp_pandas)
        filter_list.append(timestamp_filter)
    if camera_id is not None:
        camera_id_filter = (df['camera_id'] == camera_id)
        filter_list.append(camera_id_filter)
    if camera_name is not None:
        if camera_names is None:
            camera_names = process_pose_data.fetch.fetch_camera_names(
                camera_ids = df['camera_id'].unique().tolist()
            )
        camera_ids = list()
        for camera_id_in_dict, camera_name_in_dict in camera_names.items():
            if camera_name_in_dict == camera_name:
                camera_ids.append(camera_id_in_dict)
        if len(camera_ids) == 0:
            raise ValueError('No cameras match name {}'.format(camera_name))
        if len(camera_ids) > 1:
            raise ValueError('Multiple cameras match name {}'.format(camera_name))
        camera_id_from_camera_name = camera_ids[0]
        camera_id_from_camera_name_filter = (df['camera_id'] == camera_id_from_camera_name)
        filter_list.append(camera_id_from_camera_name_filter)
    if track_label is not None:
        track_label_filter = (df['track_label'] == track_label)
        filter_list.append(track_label_filter)
    combined_filter = np.bitwise_and.reduce(filter_list)
    df_selected = df.loc[combined_filter].copy()
    return df_selected

def select_pose_pair(
    df,
    pose_id_a=None,
    pose_id_b=None,
    timestamp=None,
    camera_id_a=None,
    camera_id_b=None,
    camera_name_a=None,
    camera_name_b=None,
    track_label_a=None,
    track_label_b=None,
    camera_names=None
):
    df_selected = select_pose_pairs(
        df,
        pose_id_a=pose_id_a,
        pose_id_b=pose_id_b,
        timestamp=timestamp,
        camera_id_a=camera_id_a,
        camera_id_b=camera_id_b,
        camera_name_a=camera_name_a,
        camera_name_b=camera_name_b,
        track_label_a=track_label_a,
        track_label_b=track_label_b,
        camera_names=camera_names
    )
    if len(df_selected) == 0:
        raise ValueError('No pose pairs matched criteria')
    if len(df_selected) > 1:
        raise ValueError('Multiple pose pairs matched criteria')
    return(df_selected.reset_index().iloc[0].to_dict())

def select_pose_pairs(
    df,
    pose_id_a=None,
    pose_id_b=None,
    timestamp=None,
    camera_id_a=None,
    camera_id_b=None,
    camera_name_a=None,
    camera_name_b=None,
    track_label_a=None,
    track_label_b=None,
    camera_names=None
):
    filter_list = list()
    if pose_id_a is not None:
        pose_id_a_filter = (df.index.get_level_values('pose_id_a') == pose_id_a)
        filter_list.append(pose_id_a_filter)
    if pose_id_b is not None:
        pose_id_b_filter = (df.index.get_level_values('pose_id_b') == pose_id_b)
        filter_list.append(pose_id_b_filter)
    if timestamp is not None:
        timestamp_pandas = pd.to_datetime(timestamp, utc=True)
        timestamp_filter = (df['timestamp'] == timestamp_pandas)
        filter_list.append(timestamp_filter)
    if camera_id_a is not None:
        camera_id_a_filter = (df['camera_id_a'] == camera_id_a)
        filter_list.append(camera_id_a_filter)
    if camera_id_b is not None:
        camera_id_b_filter = (df['camera_id_b'] == camera_id_b)
        filter_list.append(camera_id_b_filter)
    if camera_name_a is not None or camera_name_b is not None:
        if camera_names is None:
            camera_names = process_pose_data.fetch.fetch_camera_names(
                camera_ids = np.union1d(
                    df['camera_id_a'].unique(),
                    df['camera_id_b'].unique()
                ).tolist()
            )
    if camera_name_a is not None:
        camera_ids = list()
        for camera_id_in_dict, camera_name_in_dict in camera_names.items():
            if camera_name_in_dict == camera_name_a:
                camera_ids.append(camera_id_in_dict)
        if len(camera_ids) == 0:
            raise ValueError('No cameras match name {}'.format(camera_name_a))
        if len(camera_ids) > 1:
            raise ValueError('Multiple cameras match name {}'.format(camera_name_a))
        camera_id_from_camera_name_a = camera_ids[0]
        camera_id_from_camera_name_a_filter = (df['camera_id_a'] == camera_id_from_camera_name_a)
        filter_list.append(camera_id_from_camera_name_a_filter)
    if camera_name_b is not None:
        camera_ids = list()
        for camera_id_in_dict, camera_name_in_dict in camera_names.items():
            if camera_name_in_dict == camera_name_b:
                camera_ids.append(camera_id_in_dict)
        if len(camera_ids) == 0:
            raise ValueError('No cameras match name {}'.format(camera_name_b))
        if len(camera_ids) > 1:
            raise ValueError('Multiple cameras match name {}'.format(camera_name_b))
        camera_id_from_camera_name_b = camera_ids[0]
        camera_id_from_camera_name_b_filter = (df['camera_id_b'] == camera_id_from_camera_name_b)
        filter_list.append(camera_id_from_camera_name_b_filter)
    if track_label_a is not None:
        track_label_a_filter = (df['track_label_a'] == track_label_a)
        filter_list.append(track_label_a_filter)
    if track_label_b is not None:
        track_label_b_filter = (df['track_label_b'] == track_label_b)
        filter_list.append(track_label_b_filter)
    combined_filter = np.bitwise_and.reduce(filter_list)
    df_selected = df.loc[combined_filter].copy()
    return df_selected
