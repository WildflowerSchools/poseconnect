import cv_utils
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
            logger.info('Analyzing matches between {} and {}'.format(
                camera_name_a,
                camera_name_b
            ))
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
                    logger.info('Track {}: {} timestamps. Track {}: {} timestamps. {} common timestamps'.format(
                        track_label_a,
                        len(df_a),
                        track_label_b,
                        len(df_b),
                        len(common_timestamps)
                    ))
                    keypoints_a = np.concatenate(df_a.reindex(common_timestamps)['keypoint_array'].values)
                    keypoints_b = np.concatenate(df_b.reindex(common_timestamps)['keypoint_array'].values)
                    keypoints_a_undistorted = cv_utils.undistort_points(
                        keypoints_a,
                        camera_info[camera_device_id_a]['camera_matrix'],
                        camera_info[camera_device_id_a]['distortion_coefficients']
                    )
                    keypoints_b_undistorted = cv_utils.undistort_points(
                        keypoints_b,
                        camera_info[camera_device_id_b]['camera_matrix'],
                        camera_info[camera_device_id_b]['distortion_coefficients']
                    )
                    object_points = cv_utils.reconstruct_object_points_from_camera_poses(
                        keypoints_a_undistorted,
                        keypoints_b_undistorted,
                        camera_info[camera_device_id_a]['camera_matrix'],
                        camera_info[camera_device_id_a]['rotation_vector'],
                        camera_info[camera_device_id_a]['translation_vector'],
                        camera_info[camera_device_id_b]['rotation_vector'],
                        camera_info[camera_device_id_b]['translation_vector']
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
                    difference_a = keypoints_a_reprojected - keypoints_a
                    difference_b = keypoints_b_reprojected - keypoints_b
                    difference_a_norm = np.linalg.norm(difference_a, axis=1)
                    difference_b_norm = np.linalg.norm(difference_b, axis=1)
                    results.append({
                        'camera_device_id_a': camera_device_id_a,
                        'camera_name_a': camera_name_a,
                        'camera_device_id_b': camera_device_id_b,
                        'camera_name_b': camera_name_b,
                        'track_label_a': track_label_a,
                        'track_label_b': track_label_b,
                        'num_common_frames': num_common_frames,
                        'mean_reprojection_error_a': np.nanmean(difference_a_norm),
                        'mean_reprojection_error_b': np.nanmean(difference_b_norm)
                    })
    return(pd.DataFrame(results))
