import cv_utils
import cv2 as cv
import pandas as pd
import numpy as np
import logging
import time
import itertools

logger = logging.getLogger(__name__)

def filter_pose_tracks(
    df,
    min_pose_quality=None,
    max_pose_quality=None,
    min_keypoint_quality=None,
    max_keypoint_quality=None,
    min_num_poses_in_track=None,
    inplace=False
):
    # Make copy of input dataframe if operation is not in place
    if inplace:
        df_filtered = df
    else:
        df_filtered = df.copy()
    # Apply filters
    if min_pose_quality is not None:
        df_filtered = df_filtered.loc[df_filtered['pose_quality'] >= min_pose_quality]
    if max_pose_quality is not None:
        df_filtered = df_filtered.loc[df_filtered['pose_quality'] <= max_pose_quality]
    if min_keypoint_quality is not None or max_keypoint_quality is not None:
        df_filtered = filter_keypoint_quality(
            df=df_filtered,
            min_keypoint_quality=min_keypoint_quality,
            max_keypoint_quality=max_keypoint_quality,
            inplace=False
        )
    if min_num_poses_in_track is not None:
        df_filtered = df.groupby(['camera_device_id', 'track_label']).filter(lambda x: len(x) >= min_num_poses_in_track)
    if not inplace:
        return df_filtered

def filter_keypoint_quality(
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
    keypoint_arrays = df_filtered['keypoint_array'].values
    num_keypoint_arrays = len(keypoint_arrays)
    keypoints = np.concatenate(keypoint_arrays, axis = 0)
    keypoints_quality_arrays = df_filtered['keypoint_quality_array'].values
    num_keypoints_quality_arrays = len(keypoints_quality_arrays)
    keypoints_quality = np.concatenate(keypoints_quality_arrays)
    if num_keypoint_arrays != num_keypoints_quality_arrays:
        raise ValueError('Number of keypoint arrays ({}) does not match number of keypoint quality arrays ({})'.format(
            num_keypoint_arrays,
            num_keypoints_quality_arrays
        ))
    num_spatial_dimensions_per_keypoint = keypoints.shape[1]
    if min_keypoint_quality is not None:
        mask = np.less(
            keypoints_quality,
            min_keypoint_quality,
            where=~np.isnan(keypoints_quality)
        )
        keypoints[mask] = np.array(num_spatial_dimensions_per_keypoint*[np.nan])
        keypoints_quality[mask] = np.nan
    if max_keypoint_quality is not None:
        mask = np.greater(
            keypoints_quality,
            max_keypoint_quality,
            where=~np.isnan(keypoints_quality)
        )
        keypoints[mask] = np.array(num_spatial_dimensions_per_keypoint*[np.nan])
        keypoints_quality[mask] = np.nan
    df_filtered['keypoint_array'] = np.split(keypoints, num_keypoint_arrays)
    df_filtered['keypoint_quality_array'] = np.split(keypoints_quality, num_keypoints_quality_arrays)
    if not inplace:
        return df_filtered

def filter_num_valid_keypoints(
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
    num_keypoints = df['keypoint_quality_array'].apply(
        lambda x: np.count_nonzero(~np.isnan(x))
    )
    if min_num_keypoints is not None:
        df_filtered = df_filtered.loc[num_keypoints >= min_num_keypoints]
    if max_num_keypoints is not None:
        df_filtered = df_filtered.loc[num_keypoints <= max_num_keypoints]
    if not inplace:
        return df_filtered

def filter_pose_quality(
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

def generate_pose_pairs(
    df
):
    pose_pairs = df.groupby('timestamp').apply(generate_pose_pairs_timestamp)
    return pose_pairs

def generate_pose_pairs_timestamp(
    df
):
    df_reduced = df.reindex(columns=[
        'camera_device_id',
        'track_label',
        'keypoint_array',
        'keypoint_quality_array',
        'pose_quality'
    ])
    camera_device_ids = df_reduced['camera_device_id'].unique().tolist()
    pose_id_pairs = list()
    for camera_device_id_a, camera_device_id_b in itertools.combinations(camera_device_ids, 2):
        pose_ids_a = df_reduced.loc[df_reduced['camera_device_id'] == camera_device_id_a].index.tolist()
        pose_ids_b = df_reduced.loc[df_reduced['camera_device_id'] == camera_device_id_b].index.tolist()
        pose_id_pairs_camera_pair = list(itertools.product(pose_ids_a, pose_ids_b))
        pose_id_pairs.extend(pose_id_pairs_camera_pair)
    pose_ids_a = list()
    pose_ids_b = list()
    if len(pose_id_pairs) > 0:
        pose_ids_a, pose_ids_b = map(list, zip(*pose_id_pairs))
    pose_pairs_timestamp = pd.concat(
        (df_reduced.loc[pose_ids_a].reset_index(), df_reduced.loc[pose_ids_b].reset_index()),
        keys=['a', 'b'],
        axis=1
    )
    pose_pairs_timestamp.set_index(
        [('a', 'pose_id'), ('b', 'pose_id')],
        inplace=True
    )
    pose_pairs_timestamp.rename_axis(
        ['pose_id_a', 'pose_id_b'],
        inplace=True
    )
    return pose_pairs_timestamp

def score_pose_track_matches(
    df,
    camera_info
):
    camera_device_ids = df['camera_device_id'].unique().tolist()
    results = list()
    for camera_index_a, camera_device_id_a in enumerate(camera_device_ids):
        for camera_device_id_b in camera_device_ids[(camera_index_a + 1):]:
            camera_name_a = df.loc[df['camera_device_id'] == camera_device_id_a, 'camera_name'][0]
            camera_name_b = df.loc[df['camera_device_id'] == camera_device_id_b, 'camera_name'][0]
            track_labels_a = df.loc[df['camera_device_id'] == camera_device_id_a, 'track_label'].unique().tolist()
            track_labels_b = df.loc[df['camera_device_id'] == camera_device_id_b, 'track_label'].unique().tolist()
            num_tracks_a = len(track_labels_a)
            num_tracks_b = len(track_labels_b)
            num_potential_matches = num_tracks_a*num_tracks_b
            logger.info('Analyzing matches between {} tracks from {} and {} tracks from {} ({} potential matches)'.format(
                num_tracks_a,
                camera_name_a,
                num_tracks_b,
                camera_name_b,
                num_potential_matches
            ))
            start_time = time.time()
            for track_label_a in track_labels_a:
                for track_label_b in track_labels_b:
                    df_a = df.loc[
                        (df['camera_device_id'] == camera_device_id_a) &
                        (df['track_label'] == track_label_a)
                    ].set_index('timestamp')
                    df_b = df.loc[
                        (df['camera_device_id'] == camera_device_id_b) &
                        (df['track_label'] == track_label_b)
                    ].set_index('timestamp')
                    common_timestamps = df_a.index.intersection(df_b.index)
                    num_common_frames = len(common_timestamps)
                    if num_common_frames == 0:
                        continue
                    keypoints_a = np.concatenate(df_a.reindex(common_timestamps)['keypoint_array'].values)
                    keypoints_b = np.concatenate(df_b.reindex(common_timestamps)['keypoint_array'].values)
                    object_points = triangulate_image_points(
                        image_points_1=keypoints_a,
                        image_points_2=keypoints_b,
                        camera_matrix_1=camera_info[camera_device_id_a]['camera_matrix'],
                        distortion_coefficients_1=camera_info[camera_device_id_a]['distortion_coefficients'],
                        rotation_vector_1=camera_info[camera_device_id_a]['rotation_vector'],
                        translation_vector_1=camera_info[camera_device_id_a]['translation_vector'],
                        camera_matrix_2=camera_info[camera_device_id_b]['camera_matrix'],
                        distortion_coefficients_2=camera_info[camera_device_id_b]['distortion_coefficients'],
                        rotation_vector_2=camera_info[camera_device_id_b]['rotation_vector'],
                        translation_vector_2=camera_info[camera_device_id_b]['translation_vector']
                    )
                    keypoints_a_reprojected = cv_utils.project_points(
                        object_points,
                        camera_info[camera_device_id_a]['rotation_vector'],
                        camera_info[camera_device_id_a]['translation_vector'],
                        camera_info[camera_device_id_a]['camera_matrix'],
                        camera_info[camera_device_id_a]['distortion_coefficients']
                    )
                    keypoints_b_reprojected = cv_utils.project_points(
                        object_points,
                        camera_info[camera_device_id_b]['rotation_vector'],
                        camera_info[camera_device_id_b]['translation_vector'],
                        camera_info[camera_device_id_b]['camera_matrix'],
                        camera_info[camera_device_id_b]['distortion_coefficients']
                    )
                    diff_a = keypoints_a_reprojected - keypoints_a
                    diff_b = keypoints_b_reprojected - keypoints_b
                    diff_a_norms = np.linalg.norm(diff_a, axis=1)
                    diff_b_norms = np.linalg.norm(diff_b, axis=1)
                    diff_a_sum_squares = np.sum(np.square(diff_a), axis=1)
                    diff_b_sum_squares = np.sum(np.square(diff_b), axis=1)
                    mean_reproj_error_a = np.nanmean(diff_a_norms)
                    mean_reproj_error_b = np.nanmean(diff_b_norms)
                    root_mean_square_reproj_error_a = np.sqrt(np.nanmean(diff_a_sum_squares))
                    root_mean_square_reproj_error_b = np.sqrt(np.nanmean(diff_b_sum_squares))
                    median_reproj_error_a = np.nanmedian(diff_a_norms)
                    median_reproj_error_b = np.nanmedian(diff_b_norms)
                    root_median_square_reproj_error_a = np.sqrt(np.nanmedian(diff_a_sum_squares))
                    root_median_square_reproj_error_b = np.sqrt(np.nanmedian(diff_b_sum_squares))
                    results.append({
                        'camera_device_id_a': camera_device_id_a,
                        'camera_name_a': camera_name_a,
                        'camera_device_id_b': camera_device_id_b,
                        'camera_name_b': camera_name_b,
                        'track_label_a': track_label_a,
                        'track_label_b': track_label_b,
                        'num_common_frames': num_common_frames,
                        'mean_reproj_error_a': mean_reproj_error_a,
                        'mean_reproj_error_b': mean_reproj_error_b,
                        'root_mean_square_reproj_error_a': root_mean_square_reproj_error_a,
                        'root_mean_square_reproj_error_b': root_mean_square_reproj_error_b,
                        'median_reproj_error_a': median_reproj_error_a,
                        'median_reproj_error_b': median_reproj_error_b,
                        'root_median_square_reproj_error_a': root_median_square_reproj_error_a,
                        'root_median_square_reproj_error_b': root_median_square_reproj_error_b
                    })
            elapsed_time = time.time() - start_time
            logger.info('Scored {} potential matches in {:.3f} seconds ({:.1f} milliseconds per match)'.format(
                num_potential_matches,
                elapsed_time,
                10**3*elapsed_time/num_potential_matches
            ))
    pose_tracks_3d_scored = pd.DataFrame(results)
    pose_tracks_3d_scored.sort_values(
        [
            'camera_device_id_a',
            'camera_device_id_b',
            'track_label_a',
            'track_label_b'
        ],
        inplace=True
    )
    return pose_tracks_3d_scored

def calculate_3d_poses(
    df,
    camera_info
):
    df_list=list()
    camera_device_ids = df['camera_device_id'].unique().tolist()
    for camera_index_a, camera_device_id_a in enumerate(camera_device_ids):
        for camera_device_id_b in camera_device_ids[(camera_index_a + 1):]:
            camera_name_a = df.loc[df['camera_device_id'] == camera_device_id_a, 'camera_name'][0]
            camera_name_b = df.loc[df['camera_device_id'] == camera_device_id_b, 'camera_name'][0]
            track_labels_a = df.loc[df['camera_device_id'] == camera_device_id_a, 'track_label'].unique().tolist()
            track_labels_b = df.loc[df['camera_device_id'] == camera_device_id_b, 'track_label'].unique().tolist()
            num_tracks_a = len(track_labels_a)
            num_tracks_b = len(track_labels_b)
            num_potential_matches = num_tracks_a*num_tracks_b
            logger.info('Calculating 3D poses between {} tracks from {} and {} tracks from {} ({} potential poses)'.format(
                num_tracks_a,
                camera_name_a,
                num_tracks_b,
                camera_name_b,
                num_potential_matches
            ))
            start_time = time.time()
            for track_label_a in track_labels_a:
                for track_label_b in track_labels_b:
                    df_a = df.loc[
                        (df['camera_device_id'] == camera_device_id_a) &
                        (df['track_label'] == track_label_a),
                        ['timestamp', 'camera_device_id', 'camera_name', 'track_label', 'keypoint_array', 'keypoint_quality_array', 'pose_quality']
                    ].set_index('timestamp')
                    df_b = df.loc[
                        (df['camera_device_id'] == camera_device_id_b) &
                        (df['track_label'] == track_label_b),
                        ['timestamp', 'camera_device_id', 'camera_name', 'track_label', 'keypoint_array', 'keypoint_quality_array', 'pose_quality']
                    ].set_index('timestamp')
                    df_join = df_a.join(df_b, how='inner', lsuffix='_a', rsuffix='_b')
                    num_common_frames = len(df_join)
                    if num_common_frames == 0:
                        continue
                    keypoints_a = np.concatenate(df_join['keypoint_array_a'].values)
                    keypoints_b = np.concatenate(df_join['keypoint_array_b'].values)
                    object_points = triangulate_image_points(
                        image_points_1=keypoints_a,
                        image_points_2=keypoints_b,
                        camera_matrix_1=camera_info[camera_device_id_a]['camera_matrix'],
                        distortion_coefficients_1=camera_info[camera_device_id_a]['distortion_coefficients'],
                        rotation_vector_1=camera_info[camera_device_id_a]['rotation_vector'],
                        translation_vector_1=camera_info[camera_device_id_a]['translation_vector'],
                        camera_matrix_2=camera_info[camera_device_id_b]['camera_matrix'],
                        distortion_coefficients_2=camera_info[camera_device_id_b]['distortion_coefficients'],
                        rotation_vector_2=camera_info[camera_device_id_b]['rotation_vector'],
                        translation_vector_2=camera_info[camera_device_id_b]['translation_vector']
                    )
                    keypoints_a_reprojected = cv_utils.project_points(
                        object_points,
                        camera_info[camera_device_id_a]['rotation_vector'],
                        camera_info[camera_device_id_a]['translation_vector'],
                        camera_info[camera_device_id_a]['camera_matrix'],
                        camera_info[camera_device_id_a]['distortion_coefficients']
                    )
                    keypoints_b_reprojected = cv_utils.project_points(
                        object_points,
                        camera_info[camera_device_id_b]['rotation_vector'],
                        camera_info[camera_device_id_b]['translation_vector'],
                        camera_info[camera_device_id_b]['camera_matrix'],
                        camera_info[camera_device_id_b]['distortion_coefficients']
                    )
                    df_join['keypoint_array_3d'] = np.split(object_points, num_common_frames)
                    df_join['keypoint_array_reprojected_a'] = np.split(keypoints_a_reprojected, num_common_frames)
                    df_join['keypoint_array_reprojected_b'] = np.split(keypoints_b_reprojected, num_common_frames)
                    df_list.append(df_join)
            elapsed_time = time.time() - start_time
            logger.info('Calculated 3D poses from {} track pairs in {:.3f} seconds ({:.1f} milliseconds per pair)'.format(
                num_potential_matches,
                elapsed_time,
                10**3*elapsed_time/num_potential_matches
            ))
    poses_3d = pd.concat(df_list)
    poses_3d.reset_index(inplace=True)
    poses_3d = poses_3d.reindex(columns=[
        'camera_device_id_a',
        'camera_name_a',
        'camera_device_id_b',
        'camera_name_b',
        'track_label_a',
        'track_label_b',
        'timestamp',
        'keypoint_array_a',
        'keypoint_quality_array_a',
        'pose_quality_a',
        'keypoint_array_b',
        'keypoint_quality_array_b',
        'pose_quality_b',
        'keypoint_array_3d',
        'keypoint_array_reprojected_a',
        'keypoint_array_reprojected_b'
    ])
    poses_3d.sort_values(
        [
            'camera_device_id_a',
            'camera_device_id_b',
            'track_label_a',
            'track_label_b',
            'timestamp'
        ],
        inplace=True
    )
    return poses_3d

def score_3d_poses(
    poses_3d,
    inplace=False
):
    if inplace:
        poses_3d_scored = poses_3d
    else:
        poses_3d_scored = poses_3d.copy()
    poses_3d_scored['diff_a'] = (
        poses_3d_scored['keypoint_array_reprojected_a'] -
        poses_3d_scored['keypoint_array_a']
    )
    poses_3d_scored['diff_b'] = (
        poses_3d_scored['keypoint_array_reprojected_b'] -
        poses_3d_scored['keypoint_array_b']
    )
    poses_3d_scored['mean_reproj_error_a'] = poses_3d_scored['diff_a'].apply(
        lambda x: np.nanmean(np.linalg.norm(x, axis=1), axis=0)
    )
    poses_3d_scored['mean_reproj_error_b'] = poses_3d_scored['diff_b'].apply(
        lambda x: np.nanmean(np.linalg.norm(x, axis=1), axis=0)
    )
    poses_3d_scored['root_mean_square_reproj_error_a'] = poses_3d_scored['diff_a'].apply(
        lambda x: np.sqrt(np.nanmean(np.sum(np.square(x), axis=1), axis=0))
    )
    poses_3d_scored['root_mean_square_reproj_error_b'] = poses_3d_scored['diff_b'].apply(
        lambda x: np.sqrt(np.nanmean(np.sum(np.square(x), axis=1), axis=0))
    )
    poses_3d_scored['median_reproj_error_a'] = poses_3d_scored['diff_a'].apply(
        lambda x: np.nanmedian(np.linalg.norm(x, axis=1), axis=0)
    )
    poses_3d_scored['median_reproj_error_b'] = poses_3d_scored['diff_b'].apply(
        lambda x: np.nanmedian(np.linalg.norm(x, axis=1), axis=0)
    )
    poses_3d_scored['root_median_square_reproj_error_a'] = poses_3d_scored['diff_a'].apply(
        lambda x: np.sqrt(np.nanmedian(np.sum(np.square(x), axis=1), axis=0))
    )
    poses_3d_scored['root_median_square_reproj_error_b'] = poses_3d_scored['diff_b'].apply(
        lambda x: np.sqrt(np.nanmedian(np.sum(np.square(x), axis=1), axis=0))
    )
    poses_3d_scored.sort_values(
        [
            'camera_device_id_a',
            'camera_device_id_b',
            'track_label_a',
            'track_label_b',
            'timestamp'
        ],
        inplace=True
    )
    if not inplace:
        return poses_3d_scored

def score_3d_pose_tracks(
    poses_3d
):
    df = poses_3d.copy()
    df['diff_a'] = (
        df['keypoint_array_reprojected_a'] -
        df['keypoint_array_a']
    )
    df['diff_b'] = (
        df['keypoint_array_reprojected_b'] -
        df['keypoint_array_b']
    )
    poses_3d_grouped = df.groupby([
        'camera_device_id_a',
        'camera_name_a',
        'camera_device_id_b',
        'camera_name_b',
        'track_label_a',
        'track_label_b'
    ])
    num_common_frames = poses_3d_grouped.apply(len)
    mean_reproj_error_a = poses_3d_grouped.apply(lambda df:
        np.nanmean(np.linalg.norm(np.concatenate(df['diff_a'].values), axis=1), axis=0)
    )
    mean_reproj_error_b = poses_3d_grouped.apply(lambda df:
        np.nanmean(np.linalg.norm(np.concatenate(df['diff_b'].values), axis=1), axis=0)
    )
    root_mean_square_reproj_error_a = poses_3d_grouped.apply(lambda df:
        np.sqrt(np.nanmean(np.sum(np.square(np.concatenate(df['diff_a'].values)), axis=1), axis=0))
    )
    root_mean_square_reproj_error_b = poses_3d_grouped.apply(lambda df:
        np.sqrt(np.nanmean(np.sum(np.square(np.concatenate(df['diff_b'].values)), axis=1), axis=0))
    )
    median_reproj_error_a = poses_3d_grouped.apply(lambda df:
        np.nanmedian(np.linalg.norm(np.concatenate(df['diff_a'].values), axis=1), axis=0)
    )
    median_reproj_error_b = poses_3d_grouped.apply(lambda df:
        np.nanmedian(np.linalg.norm(np.concatenate(df['diff_b'].values), axis=1), axis=0)
    )
    root_median_square_reproj_error_a = poses_3d_grouped.apply(lambda df:
        np.sqrt(np.nanmedian(np.sum(np.square(np.concatenate(df['diff_a'].values)), axis=1), axis=0))
    )
    root_median_square_reproj_error_b = poses_3d_grouped.apply(lambda df:
        np.sqrt(np.nanmedian(np.sum(np.square(np.concatenate(df['diff_b'].values)), axis=1), axis=0))
    )
    pose_tracks_3d_scored = pd.DataFrame({
        'num_common_frames': num_common_frames,
        'mean_reproj_error_a': mean_reproj_error_a,
        'mean_reproj_error_b': mean_reproj_error_b,
        'root_mean_square_reproj_error_a': root_mean_square_reproj_error_a,
        'root_mean_square_reproj_error_b': root_mean_square_reproj_error_b,
        'median_reproj_error_a': median_reproj_error_a,
        'median_reproj_error_b': median_reproj_error_b,
        'root_median_square_reproj_error_a': root_median_square_reproj_error_a,
        'root_median_square_reproj_error_b': root_median_square_reproj_error_b
    }).reset_index()
    pose_tracks_3d_scored.sort_values(
        [
            'camera_device_id_a',
            'camera_device_id_b',
            'track_label_a',
            'track_label_b'
        ],
        inplace=True
    )
    return pose_tracks_3d_scored

def triangulate_image_points(
    image_points_1,
    image_points_2,
    camera_matrix_1,
    distortion_coefficients_1,
    rotation_vector_1,
    translation_vector_1,
    camera_matrix_2,
    distortion_coefficients_2,
    rotation_vector_2,
    translation_vector_2
):
    image_points_1 = np.asarray(image_points_1)
    image_points_2 = np.asarray(image_points_2)
    camera_matrix_1 = np.asarray(camera_matrix_1)
    distortion_coefficients_1 = np.asarray(distortion_coefficients_1)
    rotation_vector_1 = np.asarray(rotation_vector_1)
    translation_vector_1 = np.asarray(translation_vector_1)
    camera_matrix_2 = np.asarray(camera_matrix_2)
    distortion_coefficients_2 = np.asarray(distortion_coefficients_2)
    rotation_vector_2 = np.asarray(rotation_vector_2)
    translation_vector_2 = np.asarray(translation_vector_2)
    if image_points_1.size == 0 or image_points_2.size == 0:
        return np.zeros((0, 3))
    if image_points_1.shape != image_points_2.shape:
        raise ValueError('Sets of image points do not appear to be the same shape')
    image_points_shape = image_points_1.shape
    image_points_1 = image_points_1.reshape((-1, 2))
    image_points_2 = image_points_2.reshape((-1, 2))
    camera_matrix_1 = camera_matrix_1.reshape((3, 3))
    distortion_coefficients_1 = np.squeeze(distortion_coefficients_1)
    rotation_vector_1 = rotation_vector_1.reshape(3)
    translation_vector_1 = translation_vector_1.reshape(3)
    camera_matrix_2 = camera_matrix_2.reshape((3, 3))
    distortion_coefficients_2 = np.squeeze(distortion_coefficients_2)
    rotation_vector_2 = rotation_vector_2.reshape(3)
    translation_vector_2 = translation_vector_2.reshape(3)
    image_points_1_undistorted = cv_utils.undistort_points(
        image_points_1,
        camera_matrix_1,
        distortion_coefficients_1
    )
    image_points_2_undistorted = cv_utils.undistort_points(
        image_points_2,
        camera_matrix_2,
        distortion_coefficients_2
    )
    projection_matrix_1 = cv_utils.generate_projection_matrix(
        camera_matrix_1,
        rotation_vector_1,
        translation_vector_1)
    projection_matrix_2 = cv_utils.generate_projection_matrix(
        camera_matrix_2,
        rotation_vector_2,
        translation_vector_2)
    object_points_homogeneous = cv.triangulatePoints(
        projection_matrix_1,
        projection_matrix_2,
        image_points_1.T,
        image_points_2.T)
    object_points = cv.convertPointsFromHomogeneous(
        object_points_homogeneous.T
    )
    object_points = np.squeeze(object_points)
    object_points.reshape(image_points_shape[:-1] + (3,))
    return object_points

def pixel_distances(image_point_differences):
    return np.linalg.norm(image_point_differences, axis=-1)

def probability_distances(image_point_differences, pixel_distance_scale):
    return np.multiply(
        1/np.sqrt(2*np.pi*pixel_distance_scale**2),
        np.exp(
            np.divide(
                -np.square(pixel_distances(image_point_differences)),
                2*pixel_distance_scale**2
            )
        )
    )
