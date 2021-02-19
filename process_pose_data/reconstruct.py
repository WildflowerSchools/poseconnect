import process_pose_data.honeycomb_io
import cv_utils
import cv2 as cv
import pandas as pd
import numpy as np
import networkx as nx
import tqdm
import tqdm.notebook
from uuid import uuid4
import logging
import time
import itertools
from functools import partial

logger = logging.getLogger(__name__)

KEYPOINT_CATEGORIES_BY_POSE_MODEL = {
    'COCO-17': ['head', 'head', 'head', 'head', 'head', 'shoulder', 'shoulder', 'elbow', 'elbow', 'hand', 'hand', 'hip', 'hip', 'knee', 'knee', 'foot', 'foot'],
    'COCO-18': ['head', 'neck', 'shoulder', 'elbow', 'hand', 'shoulder', 'elbow', 'hand', 'hip', 'knee', 'foot', 'hip', 'knee', 'foot', 'head', 'head', 'head', 'head'],
    'MPII-15': ['head', 'neck', 'shoulder', 'elbow', 'hand', 'shoulder', 'elbow', 'hand', 'hip', 'knee', 'foot', 'hip', 'knee', 'foot', 'thorax'],
    'MPII-16': ['foot', 'knee', 'hip', 'hip', 'knee', 'foot', 'hip', 'thorax', 'neck', 'head', 'hand', 'elbow', 'shoulder', 'shoulder', 'elbow', 'hand'],
    'BODY_25': ['head', 'neck', 'shoulder', 'elbow', 'hand', 'shoulder', 'elbow', 'hand', 'hip', 'hip', 'knee', 'foot', 'hip', 'knee', 'foot', 'head', 'head', 'head', 'head', 'foot', 'foot', 'foot', 'foot', 'foot', 'foot'],
}

def reconstruct_poses_3d(
    poses_2d_df,
    pose_2d_id_column_name='pose_2d_id',
    pose_2d_ids_column_name='pose_2d_ids',
    pose_model_id=None,
    camera_calibrations=None,
    min_keypoint_quality=None,
    min_num_keypoints=None,
    min_pose_quality=None,
    min_pose_pair_score=None,
    max_pose_pair_score=25.0,
    pose_pair_score_distance_method='pixels',
    pose_pair_score_pixel_distance_scale=5.0,
    pose_pair_score_summary_method='rms',
    pose_3d_limits=None,
    room_x_limits=None,
    room_y_limits=None,
    pose_3d_graph_initial_edge_threshold=2,
    pose_3d_graph_max_dispersion=0.20,
    include_track_labels=False,
    progress_bar=False,
    notebook=False
):
    camera_ids = poses_2d_df['camera_id'].unique().tolist()
    if camera_calibrations is None:
        start = poses_2d_df['timestamp'].min().to_pydatetime()
        end = poses_2d_df['timestamp'].max().to_pydatetime()
        camera_calibrations = process_pose_data.honeycomb_io.fetch_camera_calibrations(
            camera_ids=camera_ids,
            start=start,
            end=end
        )
    missing_cameras = list()
    for camera_id in camera_ids:
        for calibration_parameter in [
            'camera_matrix',
            'distortion_coefficients',
            'rotation_vector',
            'translation_vector'
        ]:
            if camera_calibrations.get(camera_id, {}).get(calibration_parameter) is None:
                logger.warning('Camera {} in data is missing calibration information. Excluding these poses.'.format(
                    camera_id
                ))
                missing_cameras.append(camera_id)
                break
    if len(missing_cameras) > 0:
        poses_2d_df = poses_2d_df.loc[~poses_2d_df['camera_id'].isin(missing_cameras)]
    if pose_3d_limits is None:
        if room_x_limits is None or room_y_limits is None:
            raise ValueError('3D pose spatial limits no specified and room boundaries not specified')
        if pose_model_id is None:
            if 'pose_model_id' not in poses_2d_df.columns:
                raise ValueError('3D pose spatial limits not specified and pose model ID not inclued in 2D pose data')
            pose_model_ids = poses_2d_df['pose_model_id'].unique()
            if len(pose_model_ids) > 1:
                raise ValueError('Multiple pose model IDs found in 2D pose data')
            pose_model_id = pose_model_ids[0]
        pose_model = process_pose_data.honeycomb_io.fetch_pose_model_by_pose_model_id(pose_model_id)
        pose_model_name = pose_model.get('model_name')
        pose_3d_limits = pose_3d_limits_by_pose_model(
            room_x_limits=room_x_limits,
            room_y_limits=room_y_limits,
            pose_model_name=pose_model_name
        )
    reconstruct_poses_3d_timestamp_partial = partial(
        reconstruct_poses_3d_timestamp,
        camera_calibrations=camera_calibrations,
        pose_2d_id_column_name=pose_2d_id_column_name,
        pose_2d_ids_column_name=pose_2d_ids_column_name,
        min_keypoint_quality=min_keypoint_quality,
        min_num_keypoints=min_num_keypoints,
        min_pose_quality=min_pose_quality,
        min_pose_pair_score=min_pose_pair_score,
        max_pose_pair_score=max_pose_pair_score,
        pose_pair_score_distance_method=pose_pair_score_distance_method,
        pose_pair_score_pixel_distance_scale=pose_pair_score_pixel_distance_scale,
        pose_pair_score_summary_method=pose_pair_score_summary_method,
        pose_3d_limits=pose_3d_limits,
        pose_3d_graph_initial_edge_threshold=pose_3d_graph_initial_edge_threshold,
        pose_3d_graph_max_dispersion=pose_3d_graph_max_dispersion,
        include_track_labels=include_track_labels,
        validate_df=False
    )
    num_frames = len(poses_2d_df['timestamp'].unique())
    logger.info('Reconstructing 3D poses from {} 2D poses across {} frames ({} to {})'.format(
        len(poses_2d_df),
        num_frames,
        poses_2d_df['timestamp'].min().isoformat(),
        poses_2d_df['timestamp'].max().isoformat()
    ))
    start_time = time.time()
    if progress_bar:
        if notebook:
            tqdm.notebook.tqdm.pandas()
        else:
            tqdm.tqdm.pandas()
        poses_3d_df = poses_2d_df.groupby('timestamp').progress_apply(reconstruct_poses_3d_timestamp_partial)
    else:
        poses_3d_df = poses_2d_df.groupby('timestamp').apply(reconstruct_poses_3d_timestamp_partial)
    elapsed_time = time.time() - start_time
    logger.info('Generated {} 3D poses in {:.1f} seconds ({:.3f} ms/frame)'.format(
        len(poses_3d_df),
        elapsed_time,
        1000*elapsed_time/num_frames
    ))
    poses_3d_df.reset_index('timestamp', drop=True, inplace=True)
    return poses_3d_df

def pose_3d_limits_by_pose_model(
    room_x_limits,
    room_y_limits,
    pose_model_name,
    floor_z=0.0,
    foot_z_limits=(0.0, 1.0),
    knee_z_limits=(0.0, 1.0),
    hip_z_limits=(0.0, 1.5),
    thorax_z_limits=(0.0, 1.7),
    shoulder_z_limits=(0.0, 1.9),
    elbow_z_limits=(0.0, 2.0),
    hand_z_limits=(0.0, 3.0),
    neck_z_limits=(0.0, 1.9),
    head_z_limits=(0.0,2.0),
    tolerance=0.2
):
    keypoint_categories = KEYPOINT_CATEGORIES_BY_POSE_MODEL[pose_model_name]
    return pose_3d_limits(
        room_x_limits=room_x_limits,
        room_y_limits=room_y_limits,
        keypoint_categories=keypoint_categories,
        floor_z=floor_z,
        foot_z_limits=foot_z_limits,
        knee_z_limits=knee_z_limits,
        hip_z_limits=hip_z_limits,
        thorax_z_limits=thorax_z_limits,
        shoulder_z_limits=shoulder_z_limits,
        elbow_z_limits=elbow_z_limits,
        hand_z_limits=hand_z_limits,
        neck_z_limits=neck_z_limits,
        head_z_limits=head_z_limits,
        tolerance=tolerance
    )

def pose_3d_limits(
    room_x_limits,
    room_y_limits,
    keypoint_categories,
    floor_z=0.0,
    foot_z_limits=(0.0, 1.0),
    knee_z_limits=(0.0, 1.0),
    hip_z_limits=(0.0, 1.5),
    thorax_z_limits=(0.0, 1.7),
    shoulder_z_limits=(0.0, 1.9),
    elbow_z_limits=(0.0, 2.0),
    hand_z_limits=(0.0, 3.0),
    neck_z_limits=(0.0, 1.9),
    head_z_limits=(0.0,2.0),
    tolerance=0.2
):
    z_limits_dict = {
        'foot': foot_z_limits,
        'knee': knee_z_limits,
        'hip': hip_z_limits,
        'thorax': thorax_z_limits,
        'shoulder': shoulder_z_limits,
        'elbow': elbow_z_limits,
        'hand': hand_z_limits,
        'neck': neck_z_limits,
        'head': head_z_limits
    }
    pose_3d_limits_min = list()
    pose_3d_limits_max = list()
    for keypoint_category in keypoint_categories:
        pose_3d_limits_min.append([
            room_x_limits[0],
            room_y_limits[0],
            floor_z + z_limits_dict[keypoint_category][0]
        ])
        pose_3d_limits_max.append([
            room_x_limits[1],
            room_y_limits[1],
            floor_z + z_limits_dict[keypoint_category][1]
        ])
    return np.array([pose_3d_limits_min, pose_3d_limits_max]) + + np.array([[[-tolerance]], [[tolerance]]])

def reconstruct_poses_3d_timestamp(
    poses_2d_df_timestamp,
    camera_calibrations,
    pose_2d_id_column_name='pose_2d_id',
    pose_2d_ids_column_name='pose_2d_ids',
    min_keypoint_quality=None,
    min_num_keypoints=None,
    min_pose_quality=None,
    min_pose_pair_score=None,
    max_pose_pair_score=25.0,
    pose_pair_score_distance_method='pixels',
    pose_pair_score_pixel_distance_scale=5.0,
    pose_pair_score_summary_method='rms',
    pose_3d_limits=None,
    pose_3d_graph_initial_edge_threshold=2,
    pose_3d_graph_max_dispersion=0.20,
    include_track_labels=False,
    validate_df=True
):
    poses_2d_df_timestamp_copy = poses_2d_df_timestamp.copy()
    if validate_df:
        if len(poses_2d_df_timestamp_copy['timestamp'].unique()) > 1:
            raise ValueError('More than one timestamp found in data frame')
    timestamp = poses_2d_df_timestamp_copy['timestamp'][0]
    if min_keypoint_quality is not None:
        poses_2d_df_timestamp_copy = process_pose_data.filter.filter_keypoints_by_quality(
            poses_2d_df=poses_2d_df_timestamp_copy,
            min_keypoint_quality=min_keypoint_quality
        )
    poses_2d_df_timestamp_copy = process_pose_data.filter.remove_empty_2d_poses(
        poses_2d_df=poses_2d_df_timestamp_copy
    )
    if min_num_keypoints is not None:
        poses_2d_df_timestamp_copy = process_pose_data.filter.filter_poses_by_num_valid_keypoints(
            poses_2d_df=poses_2d_df_timestamp_copy,
            min_num_keypoints=min_num_keypoints
        )
    if min_pose_quality is not None:
        poses_2d_df_timestamp_copy = process_pose_data.filter.filter_poses_by_quality(
            poses_2d_df=poses_2d_df_timestamp_copy,
            min_pose_quality=min_pose_quality
        )
    pose_pairs_2d_df_timestamp = generate_pose_pairs_timestamp(
        poses_2d_df_timestamp=poses_2d_df_timestamp_copy,
        pose_2d_id_column_name=pose_2d_id_column_name
    )
    pose_pairs_2d_df_timestamp = calculate_3d_poses(
        pose_pairs_2d_df=pose_pairs_2d_df_timestamp,
        camera_calibrations=camera_calibrations
    )
    pose_pairs_2d_df_timestamp =  process_pose_data.filter.remove_empty_3d_poses(
        pose_pairs_2d_df=pose_pairs_2d_df_timestamp
    )
    pose_pairs_2d_df_timestamp =  process_pose_data.filter.remove_empty_reprojected_2d_poses(
        pose_pairs_2d_df=pose_pairs_2d_df_timestamp
    )
    pose_pairs_2d_df_timestamp = score_pose_pairs(
        pose_pairs_2d_df=pose_pairs_2d_df_timestamp,
        distance_method=pose_pair_score_distance_method,
        summary_method=pose_pair_score_summary_method,
        pixel_distance_scale=pose_pair_score_pixel_distance_scale
    )
    pose_pairs_2d_df_timestamp =  process_pose_data.filter.remove_invalid_pose_pair_scores(
        pose_pairs_2d_df=pose_pairs_2d_df_timestamp
    )
    if min_pose_pair_score is not None or max_pose_pair_score is not None:
        pose_pairs_2d_df_timestamp = process_pose_data.filter.filter_pose_pairs_by_score(
            pose_pairs_2d_df=pose_pairs_2d_df_timestamp,
            min_score=min_pose_pair_score,
            max_score=max_pose_pair_score
        )
    if pose_3d_limits is not None:
        pose_pairs_2d_df_timestamp = process_pose_data.filter.filter_pose_pairs_by_3d_pose_spatial_limits(
            pose_pairs_2d_df=pose_pairs_2d_df_timestamp,
            pose_3d_limits=pose_3d_limits
        )
    pose_pairs_2d_df_timestamp = process_pose_data.filter.filter_pose_pairs_by_best_match(
        pose_pairs_2d_df_timestamp,
        pose_2d_id_column_name=pose_2d_id_column_name
    )
    poses_3d_df_timestamp = generate_3d_poses_timestamp(
        pose_pairs_2d_df_timestamp=pose_pairs_2d_df_timestamp,
        pose_2d_ids_column_name=pose_2d_ids_column_name,
        initial_edge_threshold=pose_3d_graph_initial_edge_threshold,
        max_dispersion=pose_3d_graph_max_dispersion,
        include_track_labels=include_track_labels,
        validate_df=validate_df
    )
    if len(poses_3d_df_timestamp) == 0:
        return poses_3d_df_timestamp
    poses_3d_df_timestamp.set_index('pose_3d_id', inplace=True)
    return poses_3d_df_timestamp

def generate_pose_pairs_timestamp(
    poses_2d_df_timestamp,
    pose_2d_id_column_name='pose_2d_id'
):
    if len(poses_2d_df_timestamp) == 0:
        return pd.DataFrame()
    timestamps = poses_2d_df_timestamp['timestamp'].unique()
    if len(timestamps) > 1:
        raise ValueError('More than one timestamp in data frame')
    camera_ids = poses_2d_df_timestamp['camera_id'].unique().tolist()
    pose_2d_id_pairs = list()
    for camera_id_a, camera_id_b in itertools.combinations(camera_ids, 2):
        pose_2d_ids_a = poses_2d_df_timestamp.loc[poses_2d_df_timestamp['camera_id'] == camera_id_a].index.tolist()
        pose_2d_ids_b = poses_2d_df_timestamp.loc[poses_2d_df_timestamp['camera_id'] == camera_id_b].index.tolist()
        pose_2d_id_pairs_camera_pair = list(itertools.product(pose_2d_ids_a, pose_2d_ids_b))
        pose_2d_id_pairs.extend(pose_2d_id_pairs_camera_pair)
    pose_2d_ids_a = list()
    pose_2d_ids_b = list()
    if len(pose_2d_id_pairs) > 0:
        pose_2d_ids_a, pose_2d_ids_b = map(list, zip(*pose_2d_id_pairs))
    pose_pairs_2d_df_timestamp = pd.concat(
        (poses_2d_df_timestamp.loc[pose_2d_ids_a].reset_index(), poses_2d_df_timestamp.loc[pose_2d_ids_b].reset_index()),
        keys=['a', 'b'],
        axis=1
    )
    pose_pairs_2d_df_timestamp.set_index(
        [('a', pose_2d_id_column_name), ('b', pose_2d_id_column_name)],
        inplace=True
    )
    pose_pairs_2d_df_timestamp.rename_axis(
        [pose_2d_id_column_name + '_a', pose_2d_id_column_name + '_b'],
        inplace=True
    )
    pose_pairs_2d_df_timestamp.columns = ['{}_{}'.format(column_name[1], column_name[0]) for column_name in pose_pairs_2d_df_timestamp.columns.values]
    pose_pairs_2d_df_timestamp.rename(
        columns = {'timestamp_a': 'timestamp'},
        inplace=True
    )
    pose_pairs_2d_df_timestamp.drop(
        columns=['timestamp_b'],
        inplace=True
    )
    return pose_pairs_2d_df_timestamp

def calculate_3d_poses(
    pose_pairs_2d_df,
    camera_calibrations=None
):
    if len(pose_pairs_2d_df) == 0:
        return pose_pairs_2d_df
    if camera_calibrations is None:
        camera_ids = np.union1d(
            pose_pairs_2d_df['camera_id_a'].unique(),
            pose_pairs_2d_df['camera_id_b'].unique()
        ).tolist()
        start = pose_pairs_2d_df['timestamp'].min().to_pydatetime()
        end = pose_pairs_2d_df['timestamp'].max().to_pydatetime()
        camera_calibrations = process_pose_data.honeycomb_io.fetch_camera_calibrations(
            camera_ids=camera_ids,
            start=start,
            end=end
        )
    pose_pairs_2d_df = pose_pairs_2d_df.groupby(['camera_id_a', 'camera_id_b']).apply(
        lambda x: calculate_3d_poses_camera_pair(
            pose_pairs_2d_df_camera_pair=x,
            camera_calibrations=camera_calibrations,
            inplace=False
        )
    )
    return pose_pairs_2d_df

def calculate_3d_poses_camera_pair(
    pose_pairs_2d_df_camera_pair,
    camera_calibrations,
    inplace=False
):
    if not inplace:
        pose_pairs_2d_df_camera_pair = pose_pairs_2d_df_camera_pair.copy()
    num_pose_pairs = len(pose_pairs_2d_df_camera_pair)
    camera_ids_a = pose_pairs_2d_df_camera_pair['camera_id_a'].unique()
    camera_ids_b = pose_pairs_2d_df_camera_pair['camera_id_b'].unique()
    if len(camera_ids_a) > 1:
        raise ValueError('More than one camera ID found for camera A')
    if len(camera_ids_b) > 1:
        raise ValueError('More than one camera ID found for camera B')
    camera_id_a = camera_ids_a[0]
    camera_id_b = camera_ids_b[0]
    if camera_id_a not in camera_calibrations.keys():
        raise ValueError('Camera ID {} not found in camera calibration data'.format(
            camera_id_a
        ))
    if camera_id_b not in camera_calibrations.keys():
        raise ValueError('Camera ID {} not found in camera calibration data'.format(
            camera_id_b
        ))
    camera_calibration_a = camera_calibrations[camera_id_a]
    camera_calibration_b = camera_calibrations[camera_id_b]
    keypoint_a_lengths = pose_pairs_2d_df_camera_pair['keypoint_coordinates_2d_a'].apply(lambda x: x.shape[0]).unique()
    keypoint_b_lengths = pose_pairs_2d_df_camera_pair['keypoint_coordinates_2d_b'].apply(lambda x: x.shape[0]).unique()
    if len(keypoint_a_lengths) > 1:
        raise ValueError('Keypoint arrays in column A have differing numbers of keypoints')
    if len(keypoint_b_lengths) > 1:
        raise ValueError('Keypoint arrays in column B have differing numbers of keypoints')
    if keypoint_a_lengths[0] != keypoint_b_lengths[0]:
        raise ValueError('Keypoint arrays in column A have different number of keypoints than keypoint arrays in column B')
    keypoints_a = np.concatenate(pose_pairs_2d_df_camera_pair['keypoint_coordinates_2d_a'].values)
    keypoints_b = np.concatenate(pose_pairs_2d_df_camera_pair['keypoint_coordinates_2d_b'].values)
    keypoints_3d = triangulate_image_points(
        image_points_1=keypoints_a,
        image_points_2=keypoints_b,
        camera_matrix_1=camera_calibration_a['camera_matrix'],
        distortion_coefficients_1=camera_calibration_a['distortion_coefficients'],
        rotation_vector_1=camera_calibration_a['rotation_vector'],
        translation_vector_1=camera_calibration_a['translation_vector'],
        camera_matrix_2=camera_calibration_b['camera_matrix'],
        distortion_coefficients_2=camera_calibration_b['distortion_coefficients'],
        rotation_vector_2=camera_calibration_b['rotation_vector'],
        translation_vector_2=camera_calibration_b['translation_vector']
    )
    keypoints_a_reprojected = cv_utils.project_points(
        object_points=keypoints_3d,
        rotation_vector=camera_calibration_a['rotation_vector'],
        translation_vector=camera_calibration_a['translation_vector'],
        camera_matrix=camera_calibration_a['camera_matrix'],
        distortion_coefficients=camera_calibration_a['distortion_coefficients'],
        remove_behind_camera=True
    )
    keypoints_b_reprojected = cv_utils.project_points(
        object_points=keypoints_3d,
        rotation_vector=camera_calibration_b['rotation_vector'],
        translation_vector=camera_calibration_b['translation_vector'],
        camera_matrix=camera_calibration_b['camera_matrix'],
        distortion_coefficients=camera_calibration_b['distortion_coefficients'],
        remove_behind_camera=True
    )
    pose_pairs_2d_df_camera_pair['keypoint_coordinates_3d'] = np.split(keypoints_3d, num_pose_pairs)
    pose_pairs_2d_df_camera_pair['keypoint_coordinates_2d_a_reprojected'] = np.split(keypoints_a_reprojected, num_pose_pairs)
    pose_pairs_2d_df_camera_pair['keypoint_coordinates_2d_b_reprojected'] = np.split(keypoints_b_reprojected, num_pose_pairs)
    if not inplace:
        return pose_pairs_2d_df_camera_pair

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

def score_pose_pairs(
    pose_pairs_2d_df,
    distance_method='pixels',
    summary_method='rms',
    pixel_distance_scale=5.0
):
    if len(pose_pairs_2d_df) == 0:
        return pose_pairs_2d_df.copy()
    reprojection_difference = np.stack(
        (
            np.subtract(
                np.stack(pose_pairs_2d_df['keypoint_coordinates_2d_a_reprojected']),
                np.stack(pose_pairs_2d_df['keypoint_coordinates_2d_a'])
            ),
            np.subtract(
                np.stack(pose_pairs_2d_df['keypoint_coordinates_2d_b_reprojected']),
                np.stack(pose_pairs_2d_df['keypoint_coordinates_2d_b'])
            )
        ),
        axis=-2
    )
    if distance_method == 'pixels':
        distance = pixel_distance(reprojection_difference)
    elif distance_method == 'probability':
        distance = probability_distance(
            reprojection_difference,
            pixel_distance_scale=pixel_distance_scale
        )
    else:
        raise ValueError('Distance method not recognized')
    if summary_method == 'rms':
        score = np.sqrt(np.nanmean(np.square(distance), axis=(-1, -2)))
    elif summary_method == 'sum':
        score = np.nansum(distance, axis=(-1, -2))
    else:
        raise ValueError('Summary method not recognized')
    pose_pairs_2d_df_copy = pose_pairs_2d_df.copy()
    pose_pairs_2d_df_copy['score'] = score
    return pose_pairs_2d_df_copy

def pixel_distance(image_point_differences):
    return np.linalg.norm(image_point_differences, axis=-1)

def probability_distance(image_point_differences, pixel_distance_scale):
    return np.multiply(
        1/np.sqrt(2*np.pi*pixel_distance_scale**2),
        np.exp(
            np.divide(
                -np.square(pixel_distance(image_point_differences)),
                2*pixel_distance_scale**2
            )
        )
    )

def pose_3d_in_range(
    pose_3d,
    pose_3d_limits
):
    return np.logical_and(
        np.all(np.greater_equal(
            pose_3d,
            pose_3d_limits[0],
            out=np.full_like(pose_3d, True),
            where=(np.isfinite(pose_3d) & np.isfinite(pose_3d_limits[0]))
        )),
        np.all(np.less_equal(
            pose_3d,
            pose_3d_limits[1],
            out=np.full_like(pose_3d, True),
            where=(np.isfinite(pose_3d) & np.isfinite(pose_3d_limits[1]))
        ))
    )

def extract_best_score_indices_timestamp_camera_pair(
    pose_pairs_2d_df,
    pose_2d_id_column_name='pose_2d_id'
):
    best_a_score_for_b = pose_pairs_2d_df['score'].groupby(pose_2d_id_column_name + '_b').idxmin().dropna()
    best_b_score_for_a = pose_pairs_2d_df['score'].groupby(pose_2d_id_column_name + '_a').idxmin().dropna()
    best_score_indices = list(set(best_a_score_for_b).intersection(best_b_score_for_a))
    return best_score_indices

def generate_3d_poses_timestamp(
    pose_pairs_2d_df_timestamp,
    pose_2d_ids_column_name='pose_2d_ids',
    initial_edge_threshold=2,
    max_dispersion=0.20,
    include_track_labels=False,
    validate_df=True
):
    if len(pose_pairs_2d_df_timestamp) == 0:
        return pd.DataFrame()
    timestamps = pose_pairs_2d_df_timestamp['timestamp'].unique()
    if validate_df:
        if len(timestamps) > 1:
            raise ValueError('More than one timestamp found in data frame')
    timestamp = timestamps[0]
    pose_graph = generate_pose_graph(
        pose_pairs_2d_df_timestamp=pose_pairs_2d_df_timestamp,
        include_track_labels=include_track_labels
    )
    subgraph_list = generate_k_edge_subgraph_list_iteratively(
        pose_graph=pose_graph,
        initial_edge_threshold=initial_edge_threshold,
        max_dispersion=max_dispersion
    )
    pose_3d_ids = list()
    keypoint_coordinates_3d = list()
    pose_2d_ids = list()
    if include_track_labels:
        track_labels = list()
    for subgraph in subgraph_list:
        pose_3d_ids.append(uuid4().hex)
        keypoint_coordinates_3d_list = list()
        track_label_list = list()
        pose_2d_ids_list = list()
        for pose_2d_id_1, pose_2d_id_2, keypoint_coordinates_3d_edge in subgraph.edges(data='keypoint_coordinates_3d'):
            pose_2d_ids_list.extend([pose_2d_id_1, pose_2d_id_2])
            if include_track_labels:
                track_label_list.append((
                    subgraph.nodes[pose_2d_id_1]['camera_id'],
                    subgraph.nodes[pose_2d_id_1]['track_label_2d']
                ))
                track_label_list.append((
                    subgraph.nodes[pose_2d_id_2]['camera_id'],
                    subgraph.nodes[pose_2d_id_2]['track_label_2d']
                ))
            keypoint_coordinates_3d_list.append(keypoint_coordinates_3d_edge)
        keypoint_coordinates_3d.append(np.nanmedian(np.stack(keypoint_coordinates_3d_list), axis=0))
        pose_2d_ids.append(pose_2d_ids_list)
        if include_track_labels:
            track_labels.append(track_label_list)
    if len(pose_3d_ids) == 0:
        return pd.DataFrame()
    if include_track_labels:
        poses_3d_df_timestamp = pd.DataFrame({
            'pose_3d_id': pose_3d_ids,
            'timestamp': timestamp,
            'keypoint_coordinates_3d': keypoint_coordinates_3d,
            pose_2d_ids_column_name: pose_2d_ids,
            'track_labels_2d': track_labels
        })
    else:
        poses_3d_df_timestamp = pd.DataFrame({
            'pose_3d_id': pose_3d_ids,
            'timestamp': timestamp,
            'keypoint_coordinates_3d': keypoint_coordinates_3d,
            pose_2d_ids_column_name: pose_2d_ids
        })
    return poses_3d_df_timestamp

def generate_pose_graph(
    pose_pairs_2d_df_timestamp,
    include_track_labels=False
):
    pose_graph = nx.Graph()
    for pose_2d_ids, row in pose_pairs_2d_df_timestamp.iterrows():
        if include_track_labels:
            pose_graph.add_node(
                pose_2d_ids[0],
                track_label=row['track_label_2d_a'],
                camera_id = row['camera_id_a']
            )
            pose_graph.add_node(
                pose_2d_ids[1],
                track_label=row['track_label_2d_b'],
                camera_id = row['camera_id_b']
            )
        pose_graph.add_edge(
            *pose_2d_ids,
            keypoint_coordinates_3d=row['keypoint_coordinates_3d'],
            centroid_3d=np.nanmean(row['keypoint_coordinates_3d'], axis=0)
        )
    return pose_graph

def generate_k_edge_subgraph_list_iteratively(
    pose_graph,
    initial_edge_threshold=2,
    max_dispersion=0.20
):
    subgraph_list = list()
    for nodes in nx.k_edge_components(pose_graph, initial_edge_threshold):
        if len(nodes) < 2:
            continue
        subgraph = pose_graph.subgraph(nodes)
        if subgraph.number_of_edges() ==0:
            continue
        dispersion = pose_3d_dispersion(subgraph)
        if max_dispersion is None or dispersion <= max_dispersion:
            subgraph_list.append(subgraph)
            continue
        subgraph_list.extend(generate_k_edge_subgraph_list_iteratively(
            pose_graph=subgraph,
            initial_edge_threshold=initial_edge_threshold + 1,
            max_dispersion=max_dispersion
        ))
    return subgraph_list

def pose_3d_dispersion(pose_graph):
    return np.linalg.norm(
        np.std(
            np.stack([centroid_3d for u, v, centroid_3d in pose_graph.edges(data='centroid_3d')]),
            axis=0
        )
    )
