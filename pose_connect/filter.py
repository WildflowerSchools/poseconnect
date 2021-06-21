import process_pose_data.reconstruct
import honeycomb_io
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def filter_keypoints_by_quality(
    poses_2d_df,
    min_keypoint_quality=None,
    max_keypoint_quality=None
):
    poses_2d_df_filtered = poses_2d_df.copy()
    if len(poses_2d_df_filtered) == 0:
        return poses_2d_df_filtered
    keypoint_coordinate_arrays = poses_2d_df_filtered['keypoint_coordinates_2d'].values
    num_keypoint_coordinate_arrays = len(keypoint_coordinate_arrays)
    keypoint_coordinates = np.concatenate(keypoint_coordinate_arrays, axis = 0)
    keypoint_quality_arrays = poses_2d_df_filtered['keypoint_quality_2d'].values
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
    poses_2d_df_filtered['keypoint_coordinates_2d'] = np.split(keypoint_coordinates, num_keypoint_coordinate_arrays)
    poses_2d_df_filtered['keypoint_quality_2d'] = np.split(keypoint_quality, num_keypoint_quality_arrays)
    return poses_2d_df_filtered

def remove_empty_2d_poses(
    poses_2d_df
):
    poses_2d_df_filtered = poses_2d_df.copy()
    if len(poses_2d_df_filtered) == 0:
        return poses_2d_df_filtered
    non_empty = poses_2d_df['keypoint_coordinates_2d'].apply(
        lambda x: np.any(np.all(np.isfinite(x), axis=1))
    )
    poses_2d_df_filtered = poses_2d_df_filtered.loc[non_empty].copy()
    return poses_2d_df_filtered

def filter_poses_by_num_valid_keypoints(
    poses_2d_df,
    min_num_keypoints=None,
    max_num_keypoints=None
):
    poses_2d_df_filtered = poses_2d_df.copy()
    if len(poses_2d_df_filtered) == 0:
        return poses_2d_df_filtered
    num_keypoints = poses_2d_df['keypoint_quality_2d'].apply(
        lambda x: np.count_nonzero(~np.isnan(x))
    )
    if min_num_keypoints is not None:
        poses_2d_df_filtered = poses_2d_df_filtered.loc[num_keypoints >= min_num_keypoints]
    if max_num_keypoints is not None:
        poses_2d_df_filtered = poses_2d_df_filtered.loc[num_keypoints <= max_num_keypoints]
    return poses_2d_df_filtered

def filter_poses_by_quality(
    poses_2d_df,
    min_pose_quality=None,
    max_pose_quality=None
):
    poses_2d_df_filtered = poses_2d_df.copy()
    if len(poses_2d_df_filtered) == 0:
        return poses_2d_df_filtered
    if min_pose_quality is not None:
        poses_2d_df_filtered = poses_2d_df_filtered.loc[poses_2d_df_filtered['pose_quality_2d'] >= min_pose_quality]
    if max_pose_quality is not None:
        poses_2d_df_filtered = poses_2d_df_filtered.loc[poses_2d_df_filtered['pose_quality_2d'] <= max_pose_quality]
    return poses_2d_df_filtered

def filter_pose_pairs_by_score(
    pose_pairs_2d_df,
    min_score=None,
    max_score=None
):
    pose_pairs_2d_df_filtered = pose_pairs_2d_df.copy()
    if len(pose_pairs_2d_df_filtered) == 0:
        return pose_pairs_2d_df_filtered
    if min_score is not None:
        pose_pairs_2d_df_filtered = pose_pairs_2d_df_filtered.loc[pose_pairs_2d_df_filtered['score'] >= min_score]
    if max_score is not None:
        pose_pairs_2d_df_filtered = pose_pairs_2d_df_filtered.loc[pose_pairs_2d_df_filtered['score'] <= max_score]
    return pose_pairs_2d_df_filtered

def filter_pose_pairs_by_3d_pose_spatial_limits(
    pose_pairs_2d_df,
    pose_3d_limits
):
    if len(pose_pairs_2d_df) == 0:
        return pose_pairs_2d_df
    valid_3d_poses = pose_pairs_2d_df['keypoint_coordinates_3d'].apply(
        lambda x: process_pose_data.reconstruct.pose_3d_in_range(x, pose_3d_limits)
    )
    pose_pairs_2d_df = pose_pairs_2d_df.loc[valid_3d_poses].copy()
    return pose_pairs_2d_df

def filter_pose_pairs_by_best_match(
    pose_pairs_2d_df_timestamp,
    pose_2d_id_column_name='pose_2d_id'
):
    if len(pose_pairs_2d_df_timestamp) == 0:
        return pose_pairs_2d_df_timestamp
    pose_pairs_2d_df_timestamp.sort_index(inplace=True)
    best_score_indices = list()
    for group_name, group_df in pose_pairs_2d_df_timestamp.groupby(['camera_id_a', 'camera_id_b']):
        best_score_indices.extend(process_pose_data.reconstruct.extract_best_score_indices_timestamp_camera_pair(
            pose_pairs_2d_df=group_df,
            pose_2d_id_column_name=pose_2d_id_column_name
        ))
    pose_pairs_2d_df_timestamp = pose_pairs_2d_df_timestamp.loc[best_score_indices].copy()
    return pose_pairs_2d_df_timestamp

def remove_empty_3d_poses(
    pose_pairs_2d_df
):
    pose_pairs_2d_df_filtered = pose_pairs_2d_df.copy()
    if len(pose_pairs_2d_df_filtered) == 0:
        return pose_pairs_2d_df_filtered
    non_empty = pose_pairs_2d_df['keypoint_coordinates_3d'].apply(
        lambda x: np.any(np.all(np.isfinite(x), axis=1))
    )
    pose_pairs_2d_df_filtered = pose_pairs_2d_df_filtered.loc[non_empty].copy()
    return pose_pairs_2d_df_filtered

def remove_empty_reprojected_2d_poses(
    pose_pairs_2d_df
):
    pose_pairs_2d_df_filtered = pose_pairs_2d_df.copy()
    if len(pose_pairs_2d_df_filtered) == 0:
        return pose_pairs_2d_df_filtered
    non_empty_a = pose_pairs_2d_df['keypoint_coordinates_2d_a_reprojected'].apply(
        lambda x: np.any(np.all(np.isfinite(x), axis=1))
    )
    non_empty_b = pose_pairs_2d_df['keypoint_coordinates_2d_b_reprojected'].apply(
        lambda x: np.any(np.all(np.isfinite(x), axis=1))
    )
    pose_pairs_2d_df_filtered = pose_pairs_2d_df_filtered.loc[non_empty_a & non_empty_b].copy()
    return pose_pairs_2d_df_filtered

def remove_invalid_pose_pair_scores(
    pose_pairs_2d_df
):
    pose_pairs_2d_df_filtered = pose_pairs_2d_df.copy()
    if len(pose_pairs_2d_df_filtered) == 0:
        return pose_pairs_2d_df_filtered
    valid = ~pose_pairs_2d_df['score'].isna()
    pose_pairs_2d_df_filtered = pose_pairs_2d_df_filtered.loc[valid].copy()
    return pose_pairs_2d_df_filtered

def select_random_pose(
    poses_2d_df
):
    return poses_2d_df.sample(1).reset_index().iloc[0].to_dict()

def select_random_pose_pair(
    pose_pairs_2d_df
):
    return pose_pairs_2d_df.sample(1).reset_index().iloc[0].to_dict()

def select_random_match(
    pose_pairs_2d_df
):
    return pose_pairs_2d_df.loc[pose_pairs_2d_df['match']].sample(1).reset_index().iloc[0].to_dict()

def select_random_timestamp_camera_pair(
    pose_pairs_2d_df
):
    timestamp = np.random.choice(pose_pairs_2d_df['timestamp'].unique())
    pose_pairs_2d_df_filtered = select_pose_pairs(pose_pairs_2d_df, timestamp=timestamp)
    camera_id_a = np.random.choice(pose_pairs_2d_df_filtered['camera_id_a'].unique())
    pose_pairs_2d_df_filtered = select_pose_pairs(pose_pairs_2d_df_filtered, camera_id_a=camera_id_a)
    camera_id_b = np.random.choice(pose_pairs_2d_df_filtered['camera_id_b'].unique())
    pose_pairs_2d_df_filtered = select_pose_pairs(pose_pairs_2d_df_filtered, camera_id_b=camera_id_b)
    return pose_pairs_2d_df_filtered

def select_random_timestamp(
    pose_pairs_2d_df
):
    timestamp = np.random.choice(pose_pairs_2d_df['timestamp'].unique())
    pose_pairs_2d_df_filtered = select_pose_pairs(pose_pairs_2d_df, timestamp=timestamp)
    return pose_pairs_2d_df_filtered

def select_random_camera_pair(
    pose_pairs_2d_df
):
    camera_id_a = np.random.choice(pose_pairs_2d_df['camera_id_a'].unique())
    pose_pairs_2d_df_filtered = select_pose_pairs(pose_pairs_2d_df, camera_id_a=camera_id_a)
    camera_id_b = np.random.choice(pose_pairs_2d_df_filtered['camera_id_b'].unique())
    pose_pairs_2d_df_filtered = select_pose_pairs(pose_pairs_2d_df_filtered, camera_id_b=camera_id_b)
    return pose_pairs_2d_df_filtered

def select_pose(
    poses_2d_df,
    pose_2d_id=None,
    timestamp=None,
    camera_id=None,
    camera_name=None,
    track_label=None,
    camera_names=None
):
    poses_2d_df_selected = select_poses(
        poses_2d_df,
        pose_2d_id=pose_2d_id,
        timestamp=timestamp,
        camera_id=camera_id,
        camera_name=camera_name,
        track_label=track_label,
        camera_names=camera_names
    )
    if len(poses_2d_df_selected) == 0:
        raise ValueError('No poses matched criteria')
    if len(poses_2d_df_selected) > 1:
        raise ValueError('Multiple poses matched criteria')
    return(poses_2d_df_selected.reset_index().iloc[0].to_dict())

def select_poses(
    poses_2d_df,
    pose_2d_id=None,
    timestamp=None,
    camera_id=None,
    camera_name=None,
    track_label=None,
    camera_names=None
):
    filter_list = list()
    if pose_2d_id is not None:
        pose_2d_id_filter = (poses_2d_df.index == pose_2d_id)
        filter_list.append(pose_2d_id_filter)
    if timestamp is not None:
        timestamp_pandas = pd.to_datetime(timestamp, utc=True)
        timestamp_filter = (poses_2d_df['timestamp'] == timestamp_pandas)
        filter_list.append(timestamp_filter)
    if camera_id is not None:
        camera_id_filter = (poses_2d_df['camera_id'] == camera_id)
        filter_list.append(camera_id_filter)
    if camera_name is not None:
        if camera_names is None:
            camera_names = honeycomb_io.fetch_camera_names(
                camera_ids = poses_2d_df['camera_id'].unique().tolist()
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
        camera_id_from_camera_name_filter = (poses_2d_df['camera_id'] == camera_id_from_camera_name)
        filter_list.append(camera_id_from_camera_name_filter)
    if track_label is not None:
        track_label_filter = (poses_2d_df['track_label_2d'] == track_label)
        filter_list.append(track_label_filter)
    combined_filter = np.bitwise_and.reduce(filter_list)
    poses_2d_df_selected = poses_2d_df.loc[combined_filter].copy()
    return poses_2d_df_selected

def select_pose_pair(
    pose_pairs_2d_df,
    pose_2d_id_a=None,
    pose_2d_id_b=None,
    timestamp=None,
    camera_id_a=None,
    camera_id_b=None,
    camera_name_a=None,
    camera_name_b=None,
    track_label_a=None,
    track_label_b=None,
    camera_names=None
):
    pose_pairs_2d_df_selected = select_pose_pairs(
        pose_pairs_2d_df,
        pose_2d_id_a=pose_2d_id_a,
        pose_2d_id_b=pose_2d_id_b,
        timestamp=timestamp,
        camera_id_a=camera_id_a,
        camera_id_b=camera_id_b,
        camera_name_a=camera_name_a,
        camera_name_b=camera_name_b,
        track_label_a=track_label_a,
        track_label_b=track_label_b,
        camera_names=camera_names
    )
    if len(pose_pairs_2d_df_selected) == 0:
        raise ValueError('No pose pairs matched criteria')
    if len(pose_pairs_2d_df_selected) > 1:
        raise ValueError('Multiple pose pairs matched criteria')
    return(pose_pairs_2d_df_selected.reset_index().iloc[0].to_dict())

def select_pose_pairs(
    pose_pairs_2d_df,
    pose_2d_id_column_name='pose_2d_id',
    pose_2d_id_a=None,
    pose_2d_id_b=None,
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
    if pose_2d_id_a is not None:
        pose_2d_id_a_filter = (pose_pairs_2d_df.index.get_level_values(pose_2d_id_column_name + '_a') == pose_2d_id_a)
        filter_list.append(pose_2d_id_a_filter)
    if pose_2d_id_b is not None:
        pose_2d_id_b_filter = (pose_pairs_2d_df.index.get_level_values(pose_2d_id_column_name + '_b') == pose_2d_id_b)
        filter_list.append(pose_2d_id_b_filter)
    if timestamp is not None:
        timestamp_pandas = pd.to_datetime(timestamp, utc=True)
        timestamp_filter = (pose_pairs_2d_df['timestamp'] == timestamp_pandas)
        filter_list.append(timestamp_filter)
    if camera_id_a is not None:
        camera_id_a_filter = (pose_pairs_2d_df['camera_id_a'] == camera_id_a)
        filter_list.append(camera_id_a_filter)
    if camera_id_b is not None:
        camera_id_b_filter = (pose_pairs_2d_df['camera_id_b'] == camera_id_b)
        filter_list.append(camera_id_b_filter)
    if camera_name_a is not None or camera_name_b is not None:
        if camera_names is None:
            camera_names = honeycomb_io.fetch_camera_names(
                camera_ids = np.union1d(
                    pose_pairs_2d_df['camera_id_a'].unique(),
                    pose_pairs_2d_df['camera_id_b'].unique()
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
        camera_id_from_camera_name_a_filter = (pose_pairs_2d_df['camera_id_a'] == camera_id_from_camera_name_a)
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
        camera_id_from_camera_name_b_filter = (pose_pairs_2d_df['camera_id_b'] == camera_id_from_camera_name_b)
        filter_list.append(camera_id_from_camera_name_b_filter)
    if track_label_a is not None:
        track_label_a_filter = (pose_pairs_2d_df['track_label_2d_a'] == track_label_a)
        filter_list.append(track_label_a_filter)
    if track_label_b is not None:
        track_label_b_filter = (pose_pairs_2d_df['track_label_2d_b'] == track_label_b)
        filter_list.append(track_label_b_filter)
    combined_filter = np.bitwise_and.reduce(filter_list)
    pose_pairs_2d_df_selected = pose_pairs_2d_df.loc[combined_filter].copy()
    return pose_pairs_2d_df_selected

def filter_pose_tracks_3d(
    poses_3d_with_tracks_df,
    num_poses_min=10
):
    poses_3d_with_tracks_filtered_df = poses_3d_with_tracks_df.groupby('pose_track_3d_id').filter(lambda x: len(x) >= num_poses_min)
    return poses_3d_with_tracks_filtered_df
