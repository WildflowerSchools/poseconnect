import cv_utils
import pandas as pd
import numpy as np
import logging
import time

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
                    logger.debug('Track {}: {} timestamps. Track {}: {} timestamps. {} common timestamps'.format(
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
                    difference_a_norms = np.linalg.norm(difference_a, axis=1)
                    difference_b_norms = np.linalg.norm(difference_b, axis=1)
                    difference_a_sum_squares = np.sum(np.square(difference_a), axis=1)
                    difference_b_sum_squares = np.sum(np.square(difference_b), axis=1)
                    mean_reprojection_error_a = np.nanmean(difference_a_norms)
                    mean_reprojection_error_b = np.nanmean(difference_b_norms)
                    rms_reprojection_error_a = np.sqrt(np.nanmean(difference_a_sum_squares))
                    rms_reprojection_error_b = np.sqrt(np.nanmean(difference_b_sum_squares))
                    median_reprojection_error_a = np.nanmedian(difference_a_norms)
                    median_reprojection_error_b = np.nanmedian(difference_b_norms)
                    r_median_s_reprojection_error_a = np.sqrt(np.nanmedian(difference_a_sum_squares))
                    r_median_s_reprojection_error_b = np.sqrt(np.nanmedian(difference_b_sum_squares))
                    results.append({
                        'camera_device_id_a': camera_device_id_a,
                        'camera_name_a': camera_name_a,
                        'camera_device_id_b': camera_device_id_b,
                        'camera_name_b': camera_name_b,
                        'track_label_a': track_label_a,
                        'track_label_b': track_label_b,
                        'num_common_frames': num_common_frames,
                        'mean_reprojection_error_a': mean_reprojection_error_a,
                        'mean_reprojection_error_b': mean_reprojection_error_b,
                        'median_reprojection_error_a': median_reprojection_error_a,
                        'median_reprojection_error_b': median_reprojection_error_b,
                        'rms_reprojection_error_a': rms_reprojection_error_a,
                        'rms_reprojection_error_b': rms_reprojection_error_b,
                        'r_median_s_reprojection_error_a': r_median_s_reprojection_error_a,
                        'r_median_s_reprojection_error_b': r_median_s_reprojection_error_b
                    })
            elapsed_time = time.time() - start_time
            logger.info('Scored {} potential matches in {:.3f} seconds ({:.1f} milliseconds per match)'.format(
                num_potential_matches,
                elapsed_time,
                10**3*elapsed_time/num_potential_matches
            ))
    return(pd.DataFrame(results))
