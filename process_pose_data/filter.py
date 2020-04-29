import process_pose_data.fetch
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def filter_keypoints_by_quality(
    df,
    min_keypoint_quality=None,
    max_keypoint_quality=None,
    inplace=False
):
    # Make copy of input dataframe if operation is not in place
    if inplace:
        df_filtered = df
    else:
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
    if not inplace:
        return df_filtered

def filter_poses_by_num_valid_keypoints(
    df,
    min_num_keypoints=None,
    max_num_keypoints=None,
    inplace=False
):
    # Make copy of input dataframe if operation is not in place
    if inplace:
        df_filtered = df
    else:
        df_filtered = df.copy()
    num_keypoints = df['keypoint_quality'].apply(
        lambda x: np.count_nonzero(~np.isnan(x))
    )
    if min_num_keypoints is not None:
        df_filtered = df_filtered.loc[num_keypoints >= min_num_keypoints]
    if max_num_keypoints is not None:
        df_filtered = df_filtered.loc[num_keypoints <= max_num_keypoints]
    if not inplace:
        return df_filtered

def filter_poses_by_quality(
    df,
    min_pose_quality=None,
    max_pose_quality=None,
    inplace=False
):
    # Make copy of input dataframe if operation is not in place
    if inplace:
        df_filtered = df
    else:
        df_filtered = df.copy()
    if min_pose_quality is not None:
        df_filtered = df_filtered.loc[df_filtered['pose_quality'] >= min_pose_quality]
    if max_pose_quality is not None:
        df_filtered = df_filtered.loc[df_filtered['pose_quality'] <= max_pose_quality]
    if not inplace:
        return df_filtered

def filter_pose_pairs_by_score(
    df,
    min_score=None,
    max_score=None,
    inplace=False
):
    # Make copy of input dataframe if operation is not in place
    if inplace:
        df_filtered = df
    else:
        df_filtered = df.copy()
    if min_score is not None:
        df_filtered = df_filtered.loc[df_filtered['score'] >= min_score]
    if max_score is not None:
        df_filtered = df_filtered.loc[df_filtered['score'] <= max_score]
    if not inplace:
        return df_filtered
