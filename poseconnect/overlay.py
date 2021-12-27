import poseconnect.utils
import poseconnect.defaults
# import process_pose_data.local_io
# import poseconnect.visualize
# import honeycomb_io
# import video_io
import pandas as pd
import numpy as np
import cv_utils
import cv2 as cv
# import ffmpeg
# import matplotlib.pyplot as plt
import matplotlib.colors
# import seaborn as sns
import tqdm
# import slugify
# import functools
import datetime
# import string
import logging
# import multiprocessing
# import os

logger = logging.getLogger(__name__)

KEYPOINT_CONNECTORS_BY_POSE_MODEL = {
    'COCO-17': [[5, 6], [5, 11], [6, 12], [11, 12], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [11, 13], [13, 15], [12, 14], [14, 16]],
    'COCO-18': [[2, 5], [2, 8], [5, 11], [8, 11], [1, 2], [1, 5], [1, 0], [0, 14], [0, 15], [0, 16], [0, 17], [1, 16], [1, 17], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13]],
    'MPII-15': [[0, 1], [1, 14], [14, 2], [2, 3], [3, 4], [14, 5], [5, 6], [6, 7], [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]],
    'MPII-16': [[6, 7], [7, 8], [8, 9], [6, 2], [2, 1], [1, 0], [6, 3], [3, 4], [4, 5], [8, 12], [12, 11], [11, 10], [8, 13], [13, 14], [14, 15]],
    'BODY_25': [[2, 3], [3, 4], [5, 2], [5, 6], [6, 7], [0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [15, 17], [16, 18], [1, 8], [8, 9], [9, 10], [10, 11], [11, 24], [24, 22], [24, 23], [8, 12], [12, 13], [13, 14], [14, 21], [21, 19], [21, 20]]
}

def overlay_poses_3d_video(
    input_video_path,
    poses_3d,
    camera_calibrations,
    output_video_path=None,
    output_video_path_suffix='poses_overlay',
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_color=poseconnect.defaults.OVERLAY_POSE_LABEL_COLOR,
    pose_label_background_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BACKGROUND_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_LINE_WIDTH
):
    raise NotImplementedError('Function overlay_poses_3d_video() not implemented yet')

def overlay_poses_2d_video(
    poses_2d,
    video_input_path,
    video_start_time,
    video_fps=None,
    video_frame_count=None,
    video_output_path=None,
    video_output_path_suffix=None,
    video_output_fourcc_string=None,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_color=poseconnect.defaults.OVERLAY_POSE_LABEL_COLOR,
    pose_label_background_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BACKGROUND_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_LINE_WIDTH,
    progress_bar=poseconnect.defaults.PROGRESS_BAR,
    notebook=poseconnect.defaults.NOTEBOOK
):
    logger.info('Keypoint connector alpha in overlay_poses_2d_video(): {}'.format(keypoint_connector_alpha))
    poses_2d = poseconnect.utils.ingest_poses_2d(poses_2d)
    if poses_2d['camera_id'].nunique() > 1:
        raise ValueError('2D pose data contains multiple camera IDs for a single video')
    logger.info('Ingested {} 2D poses spanning time period {} to {}'.format(
        len(poses_2d),
        poses_2d['timestamp'].min().isoformat(),
        poses_2d['timestamp'].max().isoformat()
    ))
    logger.info('Video input path: {}'.format(video_input_path))
    if video_start_time.tzinfo is None:
        logger.info('Specified video start time is timezone-naive. Assuming UTC')
        video_start_time=video_start_time.replace(tzinfo=datetime.timezone.utc)
    video_start_time = video_start_time.astimezone(datetime.timezone.utc)
    logger.info('Video start time is specified as {}'.format(
        video_start_time.isoformat()
    ))
    video_input = cv_utils.VideoInput(
        input_path=video_input_path,
        start_time=video_start_time
    )
    if video_fps is None:
        logger.info('Video frame rate not specified. Attempting to read from video file.')
        video_fps = video_input.video_parameters.fps
        if video_fps is None:
            raise ValueError('Failed to rate video frame rate from video file.')
    logger.info('Video frame rate is {} frames per second'.format(
        video_fps
    ))
    if video_frame_count is None:
        logger.info('Video frame count not specified. Attempting to read from video file.')
        video_frame_count = video_input.video_parameters.frame_count
        if video_frame_count is None:
            raise ValueError('Failed to rate video frame count from video file.')
    logger.info('Video frame count is {}'.format(
        video_frame_count
    ))
    if video_output_path is None:
        raise NotImplementedError('Auto-generation of video output path not yet implemented')
    # video_output_path = os.path.join(
    #     output_directory,
    #     '{}_{}_{}.{}'.format(
    #         output_filename_prefix,
    #         video_timestamp.strftime(output_filename_datetime_format),
    #         slugify.slugify(camera_name),
    #         output_filename_extension
    #     )
    # )
    logger.info('Video output path: {}'.format(video_output_path))
    video_output_parameters = video_input.video_parameters
    if video_output_fourcc_string is not None:
        video_output_parameters.fourcc_int = cv_utils.fourcc_string_to_int(video_output_fourcc_string)
    video_output = cv_utils.VideoOutput(
        video_output_path,
        video_parameters=video_output_parameters
    )
    video_timestamps, aligned_pose_timestamps = align_timestamps(
        pose_timestamps=poses_2d['timestamp'],
        video_start_time=video_start_time,
        video_fps=video_fps,
        video_frame_count=video_frame_count
    )
    if progress_bar:
        if notebook:
            t = tqdm.tqdm_notebook(total=video_frame_count)
        else:
            t = tqdm.tqdm(total=video_frame_count)
    for frame_index, pose_timestamp in enumerate(aligned_pose_timestamps):
        frame = video_input.get_frame()
        if frame is None:
            raise ValueError('Input video ended unexpectedly at frame number {}'.format(frame_index))
        for pose_id, row in poses_2d.loc[poses_2d['timestamp'] == pose_timestamp].iterrows():
            frame=overlay_poses_2d_image(
                poses_2d=poses_2d.loc[poses_2d['timestamp'] == pose_timestamp],
                image=frame,
                draw_keypoint_connectors=draw_keypoint_connectors,
                keypoint_connectors=keypoint_connectors,
                pose_model_name=pose_model_name,
                pose_color=pose_color,
                keypoint_radius=keypoint_radius,
                keypoint_alpha=keypoint_alpha,
                keypoint_connector_alpha=keypoint_connector_alpha,
                keypoint_connector_linewidth=keypoint_connector_linewidth,
                pose_label_color=pose_label_color,
                pose_label_background_alpha=pose_label_background_alpha,
                pose_label_font_scale=pose_label_font_scale,
                pose_label_line_width=pose_label_line_width
            )
        video_output.write_frame(frame)
        if progress_bar:
            t.update()
    video_input.close()
    video_output.close()
    return video_timestamps, aligned_pose_timestamps

def overlay_poses_3d_image(
    poses_3d,
    camera_calibrations,
    image,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_color=poseconnect.defaults.OVERLAY_POSE_LABEL_COLOR,
    pose_label_background_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BACKGROUND_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_LINE_WIDTH
):
    raise NotImplementedError('Function overlay_poses_3d_image() not implemented yet')

def overlay_poses_2d_image(
    poses_2d,
    image,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_color=poseconnect.defaults.OVERLAY_POSE_LABEL_COLOR,
    pose_label_background_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BACKGROUND_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_LINE_WIDTH
):
    logger.info('Keypoint connector alpha in overlay_poses_2d_image(): {}'.format(keypoint_connector_alpha))
    poses_2d = poseconnect.utils.ingest_poses_2d(poses_2d)
    if poses_2d['timestamp'].nunique() > 1:
        raise ValueError('2D pose data contains multiple timestamps for a single image')
    if poses_2d['camera_id'].nunique() > 1:
        raise ValueError('2D pose data contains multiple camera IDs for a single image')
    for pose_2d_id, row in poses_2d.iterrows():
        image = overlay_pose_2d_image(
            image=image,
            keypoint_coordinates_2d=row['keypoint_coordinates_2d'],
            pose_label=row.get('pose_label'),
            draw_keypoint_connectors=draw_keypoint_connectors,
            keypoint_connectors=keypoint_connectors,
            pose_model_name=pose_model_name,
            pose_color=pose_color,
            keypoint_radius=keypoint_radius,
            keypoint_alpha=keypoint_alpha,
            keypoint_connector_alpha=keypoint_connector_alpha,
            keypoint_connector_linewidth=keypoint_connector_linewidth,
            pose_label_color=pose_label_color,
            pose_label_background_alpha=pose_label_background_alpha,
            pose_label_font_scale=pose_label_font_scale,
            pose_label_line_width=pose_label_line_width
        )
    return image

def overlay_pose_3d_image(
    keypoint_coordinates_3d,
    camera_calibration,
    image,
    pose_label=None,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_color=poseconnect.defaults.OVERLAY_POSE_LABEL_COLOR,
    pose_label_background_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BACKGROUND_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_LINE_WIDTH
):
    raise NotImplementedError('Function overlay_pose_3d_image() not implemented yet')

def overlay_pose_2d_image(
    keypoint_coordinates_2d,
    image,
    pose_label=None,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_color=poseconnect.defaults.OVERLAY_POSE_LABEL_COLOR,
    pose_label_background_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BACKGROUND_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_LINE_WIDTH
):
    logger.info('Keypoint connector alpha in overlay_pose_2d_image(): {}'.format(keypoint_connector_alpha))
    pose_color = matplotlib.colors.to_hex(pose_color, keep_alpha=False)
    logger.info('Pose color in overlay_pose_2d_image() after hex conversion: {}'.format(pose_color))
    pose_label_color = matplotlib.colors.to_hex(pose_label_color, keep_alpha=False)
    if draw_keypoint_connectors and keypoint_connectors is None and pose_model_name is not None:
        keypoint_connectors = KEYPOINT_CONNECTORS_BY_POSE_MODEL[pose_model_name]
    keypoint_coordinates_2d = np.asarray(keypoint_coordinates_2d).reshape((-1, 2))
    if not np.any(np.all(np.isfinite(keypoint_coordinates_2d), axis=1), axis=0):
        return image
    valid_keypoints = np.all(np.isfinite(keypoint_coordinates_2d), axis=1)
    plottable_points = keypoint_coordinates_2d[valid_keypoints]
    new_image = image
    for point_index in range(plottable_points.shape[0]):
        new_image = cv_utils.draw_circle(
            original_image=new_image,
            coordinates=plottable_points[point_index],
            radius=keypoint_radius,
            line_width=1,
            color=pose_color,
            fill=True,
            alpha=keypoint_alpha
        )
    if draw_keypoint_connectors and (keypoint_connectors is not None):
        for keypoint_connector in keypoint_connectors:
            keypoint_from_index = keypoint_connector[0]
            keypoint_to_index = keypoint_connector[1]
            if valid_keypoints[keypoint_from_index] and valid_keypoints[keypoint_to_index]:
                new_image=cv_utils.draw_line(
                    original_image=new_image,
                    coordinates=[
                        keypoint_coordinates_2d[keypoint_from_index],
                        keypoint_coordinates_2d[keypoint_to_index]
                    ],
                    line_width=keypoint_connector_linewidth,
                    color=pose_color,
                    alpha=keypoint_connector_alpha
                )
    if pd.notna(pose_label):
        pose_label_anchor = np.nanmean(keypoint_coordinates_2d, axis=0)
        text_box_size, baseline = cv.getTextSize(
            text=str(pose_label),
            fontFace=cv.FONT_HERSHEY_PLAIN,
            fontScale=pose_label_font_scale,
            thickness=pose_label_line_width
        )
        new_image=cv_utils.draw_rectangle(
            original_image=new_image,
            coordinates=[
                [
                    pose_label_anchor[0] - text_box_size[0]/2,
                    pose_label_anchor[1] - (text_box_size[1] + baseline)/2
                ],
                [
                    pose_label_anchor[0] + text_box_size[0]/2,
                    pose_label_anchor[1] + (text_box_size[1] + baseline)/2
                ]
            ],
            line_width=1.5,
            color=pose_color,
            fill=True,
            alpha=pose_label_background_alpha
        )
        new_image=cv_utils.draw_text(
            original_image=new_image,
            coordinates=pose_label_anchor,
            text=str(pose_label),
            horizontal_alignment='center',
            vertical_alignment='middle',
            font_face=cv.FONT_HERSHEY_PLAIN,
            font_scale=pose_label_font_scale,
            line_width=pose_label_line_width,
            color=pose_label_color
        )
    return new_image

def align_timestamps(
    pose_timestamps,
    video_start_time,
    video_fps,
    video_frame_count
):
    pose_timestamps = pd.DatetimeIndex(
        pd.to_datetime(pose_timestamps, utc=True)
        .drop_duplicates()
        .sort_values()
    )
    if video_start_time.tzinfo is None:
        logger.info('Specified video start time is timezone-naive. Assuming UTC')
        video_start_time=video_start_time.replace(tzinfo=datetime.timezone.utc)
    video_start_time = video_start_time.astimezone(datetime.timezone.utc)
    frame_period_microseconds = 10**6/video_fps
    video_timestamps = pd.date_range(
        start=video_start_time,
        freq=pd.tseries.offsets.DateOffset(microseconds=frame_period_microseconds),
        periods=video_frame_count
    )
    aligned_pose_timestamps = list()
    for video_timestamp in video_timestamps:
        nearby_pose_timestamps = pose_timestamps[
            (pose_timestamps >= video_timestamp - datetime.timedelta(microseconds=frame_period_microseconds)/2) &
            (pose_timestamps < video_timestamp + datetime.timedelta(microseconds=frame_period_microseconds)/2)
        ]
        if len(nearby_pose_timestamps) == 0:
            logger.info('There are no pose timestamps nearby video timestamp {}.'.format(
                video_timestamp.isoformat()
            ))
            aligned_pose_timestamp = None
        elif len(nearby_pose_timestamps) == 1:
            aligned_pose_timestamp = nearby_pose_timestamps[0]
        else:
            time_distances = [
                max(nearby_pose_timestamp.to_pydatetime(), video_timestamp.to_pydatetime()) -
                min(nearby_pose_timestamp.to_pydatetime(), video_timestamp.to_pydatetime())
                for nearby_pose_timestamp in nearby_pose_timestamps
            ]
            aligned_pose_timestamp = nearby_pose_timestamps[np.argmin(time_distances)]
            logger.info('There are {} pose timestamps nearby video timestamp {}: {}. Chose pose timestamp {}'.format(
                len(nearby_pose_timestamps),
                video_timestamp.isoformat(),
                [nearby_pose_timestamp.isoformat() for nearby_pose_timestamp in nearby_pose_timestamps],
                aligned_pose_timestamp.isoformat()
            ))
        aligned_pose_timestamps.append(aligned_pose_timestamp)
    aligned_pose_timestamps = pd.DatetimeIndex(aligned_pose_timestamps)
    return video_timestamps, aligned_pose_timestamps
