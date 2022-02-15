import poseconnect.utils
import poseconnect.defaults
# import process_pose_data.local_io
# import poseconnect.visualize
# import honeycomb_io
# import video_io
import pandas as pd
import numpy as np
import cv_utils
# import cv2 as cv
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
import os

logger = logging.getLogger(__name__)

KEYPOINT_CONNECTORS_BY_POSE_MODEL = {
    'COCO-17': [[5, 6], [5, 11], [6, 12], [11, 12], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [11, 13], [13, 15], [12, 14], [14, 16]],
    'COCO-18': [[2, 5], [2, 8], [5, 11], [8, 11], [1, 2], [1, 5], [1, 0], [0, 14], [0, 15], [0, 16], [0, 17], [1, 16], [1, 17], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13]],
    'MPII-15': [[0, 1], [1, 14], [14, 2], [2, 3], [3, 4], [14, 5], [5, 6], [6, 7], [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]],
    'MPII-16': [[6, 7], [7, 8], [8, 9], [6, 2], [2, 1], [1, 0], [6, 3], [3, 4], [4, 5], [8, 12], [12, 11], [11, 10], [8, 13], [13, 14], [14, 15]],
    'BODY_25': [[2, 3], [3, 4], [5, 2], [5, 6], [6, 7], [0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [15, 17], [16, 18], [1, 8], [8, 9], [9, 10], [10, 11], [11, 24], [24, 22], [24, 23], [8, 12], [12, 13], [13, 14], [14, 21], [21, 19], [21, 20]]
}

def overlay_poses_2d(
    poses_2d,
    camera_id,
    video_input_path,
    video_start_time,
    video_fps=poseconnect.defaults.OVERLAY_VIDEO_FPS,
    video_frame_count=poseconnect.defaults.OVERLAY_VIDEO_FRAME_COUNT,
    video_output_path=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_PATH,
    video_output_directory=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_DIRECTORY,
    video_output_filename_suffix=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FILENAME_SUFFIX_POSES_2D,
    video_output_filename_extension=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FILENAME_EXTENSION,
    video_output_fourcc_string=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FOURCC_STRING,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_text_color=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_COLOR,
    pose_label_box_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BOX_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_text_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_LINE_WIDTH,
    progress_bar=poseconnect.defaults.PROGRESS_BAR,
    notebook=poseconnect.defaults.NOTEBOOK
):
    overlay_poses_video(
        poses=poses_2d,
        video_input_path=video_input_path,
        video_start_time=video_start_time,
        pose_type='2d',
        camera_id=camera_id,
        camera_calibration=None,
        camera_calibrations=None,
        pose_label_column=None,
        pose_label_map=None,
        generate_pose_label_map=False,
        video_fps=video_fps,
        video_frame_count=video_frame_count,
        video_output_path=video_output_path,
        video_output_directory=video_output_directory,
        video_output_filename_suffix=video_output_filename_suffix,
        video_output_filename_extension=video_output_filename_extension,
        video_output_fourcc_string=video_output_fourcc_string,
        draw_keypoint_connectors=draw_keypoint_connectors,
        keypoint_connectors=keypoint_connectors,
        pose_model_name=pose_model_name,
        pose_color=pose_color,
        keypoint_radius=keypoint_radius,
        keypoint_alpha=keypoint_alpha,
        keypoint_connector_alpha=keypoint_connector_alpha,
        keypoint_connector_linewidth=keypoint_connector_linewidth,
        pose_label_text_color=pose_label_text_color,
        pose_label_box_alpha=pose_label_box_alpha,
        pose_label_font_scale=pose_label_font_scale,
        pose_label_text_line_width=pose_label_text_line_width,
        progress_bar=progress_bar,
        notebook=notebook
    )

def overlay_poses_3d(
    poses_3d,
    camera_calibrations,
    camera_id,
    video_input_path,
    video_start_time,
    video_fps=poseconnect.defaults.OVERLAY_VIDEO_FPS,
    video_frame_count=poseconnect.defaults.OVERLAY_VIDEO_FRAME_COUNT,
    video_output_path=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_PATH,
    video_output_directory=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_DIRECTORY,
    video_output_filename_suffix=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FILENAME_SUFFIX_POSES_3D,
    video_output_filename_extension=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FILENAME_EXTENSION,
    video_output_fourcc_string=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FOURCC_STRING,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_text_color=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_COLOR,
    pose_label_box_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BOX_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_text_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_LINE_WIDTH,
    progress_bar=poseconnect.defaults.PROGRESS_BAR,
    notebook=poseconnect.defaults.NOTEBOOK
):
    overlay_poses_video(
        poses=poses_3d,
        video_input_path=video_input_path,
        video_start_time=video_start_time,
        pose_type='3d',
        camera_id=camera_id,
        camera_calibration=None,
        camera_calibrations=camera_calibrations,
        pose_label_column=None,
        pose_label_map=None,
        generate_pose_label_map=False,
        video_fps=video_fps,
        video_frame_count=video_frame_count,
        video_output_path=video_output_path,
        video_output_directory=video_output_directory,
        video_output_filename_suffix=video_output_filename_suffix,
        video_output_filename_extension=video_output_filename_extension,
        video_output_fourcc_string=video_output_fourcc_string,
        draw_keypoint_connectors=draw_keypoint_connectors,
        keypoint_connectors=keypoint_connectors,
        pose_model_name=pose_model_name,
        pose_color=pose_color,
        keypoint_radius=keypoint_radius,
        keypoint_alpha=keypoint_alpha,
        keypoint_connector_alpha=keypoint_connector_alpha,
        keypoint_connector_linewidth=keypoint_connector_linewidth,
        pose_label_text_color=pose_label_text_color,
        pose_label_box_alpha=pose_label_box_alpha,
        pose_label_font_scale=pose_label_font_scale,
        pose_label_text_line_width=pose_label_text_line_width,
        progress_bar=progress_bar,
        notebook=notebook
    )

def overlay_poses_3d_tracked(
    poses_3d,
    camera_calibrations,
    camera_id,
    video_input_path,
    video_start_time,
    video_fps=poseconnect.defaults.OVERLAY_VIDEO_FPS,
    video_frame_count=poseconnect.defaults.OVERLAY_VIDEO_FRAME_COUNT,
    video_output_path=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_PATH,
    video_output_directory=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_DIRECTORY,
    video_output_filename_suffix=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FILENAME_SUFFIX_POSES_3D_TRACKED,
    video_output_filename_extension=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FILENAME_EXTENSION,
    video_output_fourcc_string=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FOURCC_STRING,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_text_color=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_COLOR,
    pose_label_box_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BOX_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_text_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_LINE_WIDTH,
    progress_bar=poseconnect.defaults.PROGRESS_BAR,
    notebook=poseconnect.defaults.NOTEBOOK
):
    overlay_poses_video(
        poses=poses_3d,
        video_input_path=video_input_path,
        video_start_time=video_start_time,
        pose_type='3d',
        camera_id=camera_id,
        camera_calibration=None,
        camera_calibrations=camera_calibrations,
        pose_label_column='pose_track_3d_id',
        pose_label_map=None,
        generate_pose_label_map=True,
        video_fps=video_fps,
        video_frame_count=video_frame_count,
        video_output_path=video_output_path,
        video_output_directory=video_output_directory,
        video_output_filename_suffix=video_output_filename_suffix,
        video_output_filename_extension=video_output_filename_extension,
        video_output_fourcc_string=video_output_fourcc_string,
        draw_keypoint_connectors=draw_keypoint_connectors,
        keypoint_connectors=keypoint_connectors,
        pose_model_name=pose_model_name,
        pose_color=pose_color,
        keypoint_radius=keypoint_radius,
        keypoint_alpha=keypoint_alpha,
        keypoint_connector_alpha=keypoint_connector_alpha,
        keypoint_connector_linewidth=keypoint_connector_linewidth,
        pose_label_text_color=pose_label_text_color,
        pose_label_box_alpha=pose_label_box_alpha,
        pose_label_font_scale=pose_label_font_scale,
        pose_label_text_line_width=pose_label_text_line_width,
        progress_bar=progress_bar,
        notebook=notebook
    )

def overlay_poses_3d_identified(
    poses_3d,
    camera_calibrations,
    camera_id,
    video_input_path,
    video_start_time,
    person_lookup=None,
    video_fps=poseconnect.defaults.OVERLAY_VIDEO_FPS,
    video_frame_count=poseconnect.defaults.OVERLAY_VIDEO_FRAME_COUNT,
    video_output_path=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_PATH,
    video_output_directory=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_DIRECTORY,
    video_output_filename_suffix=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FILENAME_SUFFIX_POSES_3D_IDENTIFIED,
    video_output_filename_extension=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FILENAME_EXTENSION,
    video_output_fourcc_string=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FOURCC_STRING,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_text_color=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_COLOR,
    pose_label_box_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BOX_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_text_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_LINE_WIDTH,
    progress_bar=poseconnect.defaults.PROGRESS_BAR,
    notebook=poseconnect.defaults.NOTEBOOK
):
    overlay_poses_video(
        poses=poses_3d,
        video_input_path=video_input_path,
        video_start_time=video_start_time,
        pose_type='3d',
        camera_id=camera_id,
        camera_calibration=None,
        camera_calibrations=camera_calibrations,
        pose_label_column='person_id',
        pose_label_map=person_lookup,
        generate_pose_label_map=True,
        video_fps=video_fps,
        video_frame_count=video_frame_count,
        video_output_path=video_output_path,
        video_output_directory=video_output_directory,
        video_output_filename_suffix=video_output_filename_suffix,
        video_output_filename_extension=video_output_filename_extension,
        video_output_fourcc_string=video_output_fourcc_string,
        draw_keypoint_connectors=draw_keypoint_connectors,
        keypoint_connectors=keypoint_connectors,
        pose_model_name=pose_model_name,
        pose_color=pose_color,
        keypoint_radius=keypoint_radius,
        keypoint_alpha=keypoint_alpha,
        keypoint_connector_alpha=keypoint_connector_alpha,
        keypoint_connector_linewidth=keypoint_connector_linewidth,
        pose_label_text_color=pose_label_text_color,
        pose_label_box_alpha=pose_label_box_alpha,
        pose_label_font_scale=pose_label_font_scale,
        pose_label_text_line_width=pose_label_text_line_width,
        progress_bar=progress_bar,
        notebook=notebook
    )

def overlay_poses_video(
    poses,
    video_input_path,
    video_start_time,
    pose_type=poseconnect.defaults.OVERLAY_POSE_TYPE,
    camera_id=poseconnect.defaults.OVERLAY_CAMERA_ID,
    camera_calibration=poseconnect.defaults.OVERLAY_CAMERA_CALIBRATION,
    camera_calibrations=poseconnect.defaults.OVERLAY_CAMERA_CALIBRATIONS,
    pose_label_column=poseconnect.defaults.OVERLAY_POSE_LABEL_COLUMN,
    pose_label_map=poseconnect.defaults.OVERLAY_POSE_LABEL_MAP,
    generate_pose_label_map=poseconnect.defaults.OVERLAY_GENERATE_POSE_LABEL_MAP,
    video_fps=poseconnect.defaults.OVERLAY_VIDEO_FPS,
    video_frame_count=poseconnect.defaults.OVERLAY_VIDEO_FRAME_COUNT,
    video_output_path=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_PATH,
    video_output_directory=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_DIRECTORY,
    video_output_filename_suffix=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FILENAME_SUFFIX,
    video_output_filename_extension=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FILENAME_EXTENSION,
    video_output_fourcc_string=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FOURCC_STRING,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_text_color=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_COLOR,
    pose_label_box_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BOX_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_text_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_LINE_WIDTH,
    draw_timestamp=poseconnect.defaults.OVERLAY_DRAW_TIMESTAMP,
    timestamp_padding=poseconnect.defaults.OVERLAY_TIMESTAMP_PADDING,
    timestamp_font_scale=poseconnect.defaults.OVERLAY_TIMESTAMP_FONT_SCALE,
    timestamp_text_line_width=poseconnect.defaults.OVERLAY_TIMESTAMP_TEXT_LINE_WIDTH,
    timestamp_text_color=poseconnect.defaults.OVERLAY_TIMESTAMP_TEXT_COLOR,
    timestamp_box_color=poseconnect.defaults.OVERLAY_TIMESTAMP_BOX_COLOR,
    timestamp_box_alpha=poseconnect.defaults.OVERLAY_TIMESTAMP_BOX_ALPHA,
    progress_bar=poseconnect.defaults.PROGRESS_BAR,
    notebook=poseconnect.defaults.NOTEBOOK
):
    poses = poseconnect.utils.ingest_poses_generic(
        data_object=poses,
        pose_type=pose_type
    )
    poses=poses.copy()
    video_start_time = poseconnect.utils.convert_to_datetime_utc(video_start_time)
    if camera_calibration is None:
        if camera_calibrations is not None and camera_id is not None:
            camera_calibrations = poseconnect.utils.ingest_camera_calibrations(camera_calibrations)
            camera_calibrations = camera_calibrations.to_dict(orient='index')
            if camera_id not in camera_calibrations.keys():
                raise ValueError('Camera ID \'{}\' not found in camera calibration data'.format(
                    camera_id
                ))
            camera_calibration = camera_calibrations.get(camera_id)
    if pose_label_column is not None:
        if pose_label_column not in poses.columns:
            raise ValueError('Specified pose label column \'{}\' not found in pose data'.format(
                pose_label_column
            ))
        if pose_label_map is not None:
            pose_label_map = poseconnect.utils.convert_to_lookup_dict(pose_label_map)
        elif generate_pose_label_map:
            source_labels = (
                poses
                .sort_values('timestamp')
                .loc[:, pose_label_column]
                .dropna()
                .drop_duplicates()
                .tolist()
            )
            target_labels = list(range(len(source_labels)))
            pose_label_map = dict(zip(source_labels, target_labels))
        if pose_label_map is not None:
            poses[pose_label_column] = poses[pose_label_column].apply(
                lambda x: pose_label_map.get(x)
            )
    timestamp_text_color = matplotlib.colors.to_hex(timestamp_text_color, keep_alpha=False)
    timestamp_box_color = matplotlib.colors.to_hex(timestamp_box_color, keep_alpha=False)
    if pose_type == '2d' and 'camera_id' in poses.columns:
        if camera_id is not None:
            poses = poses.loc[poses['camera_id'] == camera_id].copy()
        if poses['camera_id'].nunique() > 1:
            raise ValueError('Pose data contains multiple camera IDs for a single video')
    elif pose_type == '3d':
        if camera_calibration is None:
            raise ValueError('Camera calibration parameters must be specified to overlay 3D poses')
    else:
        raise ValueError('Pose type must be either \'2d\' or \'3d\'')
    logger.info('Ingested {} poses spanning time period {} to {}'.format(
        len(poses),
        poses['timestamp'].min().isoformat(),
        poses['timestamp'].max().isoformat()
    ))
    logger.info('Video input path: {}'.format(video_input_path))
    logger.info('Video start time: {}'.format(
        video_start_time.isoformat()
    ))
    video_input = cv_utils.VideoInput(
        input_path=video_input_path,
        start_time=video_start_time
    )
    if video_fps is None:
        video_fps = video_input.video_parameters.fps
        if video_fps is None:
            raise ValueError('Failed to rate video frame rate from video file.')
    logger.info('Video frame rate: {} frames per second'.format(
        video_fps
    ))
    if video_frame_count is None:
        video_frame_count = video_input.video_parameters.frame_count
        if video_frame_count is None:
            raise ValueError('Failed to rate video frame count from video file.')
    logger.info('Video frame count: {}'.format(
        video_frame_count
    ))
    logger.info('Video input path: {}'.format(video_input_path))
    if video_output_path is None:
        video_input_directory, video_input_filename = os.path.split(video_input_path)
        video_input_filename_stem, video_input_filename_extension = os.path.splitext(video_input_filename)
        if video_input_filename_extension is not None:
            video_input_filename_extension = video_input_filename_extension[1:]
        logger.info('Video input directory: {}'.format(video_input_directory))
        logger.info('Video input filename stem: {}'.format(video_input_filename_stem))
        logger.info('Video input filename extension: {}'.format(video_input_filename_extension))
        if video_output_directory is None:
            video_output_directory = video_input_directory
        if video_output_filename_suffix is None:
            video_output_filename_stem = video_input_filename_stem
        else:
            video_output_filename_stem = '_'.join([
                video_input_filename_stem,
                video_output_filename_suffix
            ])
        if video_output_filename_extension is None:
            video_output_filename_extension = video_input_filename_extension
        logger.info('Video output directory: {}'.format(video_output_directory))
        logger.info('Video output filename stem: {}'.format(video_output_filename_stem))
        logger.info('Video output filename extension: {}'.format(video_output_filename_extension))
        video_output_path = os.path.join(
            video_output_directory,
            '.'.join([
                video_output_filename_stem,
                video_output_filename_extension
            ])
        )
    logger.info('Video output path: {}'.format(video_output_path))
    video_input_parameters = video_input.video_parameters
    logger.info('Video input FOURCC integer: {}'.format(
        video_input_parameters.fourcc_int
    ))
    logger.info('Video input FOURCC string: {}'.format(
        cv_utils.fourcc_int_to_string(video_input_parameters.fourcc_int)
    ))
    video_output_parameters = video_input_parameters
    if video_output_fourcc_string is not None:
        video_output_parameters.fourcc_int = cv_utils.fourcc_string_to_int(video_output_fourcc_string)
    logger.info('Video output FOURCC integer: {}'.format(
        video_output_parameters.fourcc_int
    ))
    logger.info('Video output FOURCC string: {}'.format(
        cv_utils.fourcc_int_to_string(video_output_parameters.fourcc_int)
    ))
    video_output = cv_utils.VideoOutput(
        video_output_path,
        video_parameters=video_output_parameters
    )
    video_timestamps, aligned_pose_timestamps = align_timestamps(
        pose_timestamps=poses['timestamp'],
        video_start_time=video_start_time,
        video_fps=video_fps,
        video_frame_count=video_frame_count
    )
    logger.info('{}/{} video timestamps have aligned pose data'.format(
        aligned_pose_timestamps.notna().sum(),
        video_timestamps.notna().sum()
    ))
    if progress_bar:
        if notebook:
            t = tqdm.tqdm_notebook(total=video_frame_count)
        else:
            t = tqdm.tqdm(total=video_frame_count)
    for frame_index, video_timestamp in enumerate(video_timestamps):
        pose_timestamp = aligned_pose_timestamps[frame_index]
        frame = video_input.get_frame()
        if frame is None:
            logger.warning('Input video ended unexpectedly at frame number {}'.format(frame_index))
            break
        frame=overlay_poses_image(
            poses=poses.loc[poses['timestamp'] == pose_timestamp].copy(),
            image=frame,
            pose_type=pose_type,
            camera_calibration=camera_calibration,
            pose_label_column=pose_label_column,
            draw_keypoint_connectors=draw_keypoint_connectors,
            keypoint_connectors=keypoint_connectors,
            pose_model_name=pose_model_name,
            pose_color=pose_color,
            keypoint_radius=keypoint_radius,
            keypoint_alpha=keypoint_alpha,
            keypoint_connector_alpha=keypoint_connector_alpha,
            keypoint_connector_linewidth=keypoint_connector_linewidth,
            pose_label_text_color=pose_label_text_color,
            pose_label_box_alpha=pose_label_box_alpha,
            pose_label_font_scale=pose_label_font_scale,
            pose_label_text_line_width=pose_label_text_line_width
        )
        if draw_timestamp:
            frame = cv_utils.draw_timestamp(
                original_image=frame,
                timestamp=video_timestamp,
                padding=timestamp_padding,
                font_scale=timestamp_font_scale,
                text_line_width=timestamp_text_line_width,
                text_color=timestamp_text_color,
                text_alpha=1.0,
                box_line_width=0,
                box_color=timestamp_box_color,
                box_fill=True,
                box_alpha=timestamp_box_alpha
            )
        video_output.write_frame(frame)
        if progress_bar:
            t.update()
    video_input.close()
    video_output.close()

def overlay_poses_video_frame(
    poses,
    video_input_path,
    video_start_time,
    timestamp,
    timestamp_offset_max_milliseconds=poseconnect.defaults.OVERLAY_TIMESTAMP_OFFSET_MAX_MILLISECONDS,
    pose_type=poseconnect.defaults.OVERLAY_POSE_TYPE,
    camera_id=poseconnect.defaults.OVERLAY_CAMERA_ID,
    camera_calibration=poseconnect.defaults.OVERLAY_CAMERA_CALIBRATION,
    camera_calibrations=poseconnect.defaults.OVERLAY_CAMERA_CALIBRATIONS,
    pose_label_column=poseconnect.defaults.OVERLAY_POSE_LABEL_COLUMN,
    pose_label_map=poseconnect.defaults.OVERLAY_POSE_LABEL_MAP,
    generate_pose_label_map=poseconnect.defaults.OVERLAY_GENERATE_POSE_LABEL_MAP,
    video_fps=poseconnect.defaults.OVERLAY_VIDEO_FPS,
    video_frame_count=poseconnect.defaults.OVERLAY_VIDEO_FRAME_COUNT,
    image_output_path=poseconnect.defaults.OVERLAY_IMAGE_OUTPUT_PATH,
    image_output_directory=poseconnect.defaults.OVERLAY_IMAGE_OUTPUT_DIRECTORY,
    image_output_filename_suffix=poseconnect.defaults.OVERLAY_IMAGE_OUTPUT_FILENAME_SUFFIX,
    image_output_filename_extension=poseconnect.defaults.OVERLAY_IMAGE_OUTPUT_FILENAME_EXTENSION,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_text_color=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_COLOR,
    pose_label_box_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BOX_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_text_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_LINE_WIDTH,
    draw_timestamp=poseconnect.defaults.OVERLAY_DRAW_TIMESTAMP,
    timestamp_padding=poseconnect.defaults.OVERLAY_TIMESTAMP_PADDING,
    timestamp_font_scale=poseconnect.defaults.OVERLAY_TIMESTAMP_FONT_SCALE,
    timestamp_text_line_width=poseconnect.defaults.OVERLAY_TIMESTAMP_TEXT_LINE_WIDTH,
    timestamp_text_color=poseconnect.defaults.OVERLAY_TIMESTAMP_TEXT_COLOR,
    timestamp_box_color=poseconnect.defaults.OVERLAY_TIMESTAMP_BOX_COLOR,
    timestamp_box_alpha=poseconnect.defaults.OVERLAY_TIMESTAMP_BOX_ALPHA
):
    poses = poseconnect.utils.ingest_poses_generic(
        data_object=poses,
        pose_type=pose_type
    )
    poses=poses.copy()
    video_start_time = poseconnect.utils.convert_to_datetime_utc(video_start_time)
    timestamp = poseconnect.utils.convert_to_datetime_utc(timestamp)
    if camera_calibration is None:
        if camera_calibrations is not None and camera_id is not None:
            camera_calibrations = poseconnect.utils.ingest_camera_calibrations(camera_calibrations)
            camera_calibrations = camera_calibrations.to_dict(orient='index')
            if camera_id not in camera_calibrations.keys():
                raise ValueError('Camera ID \'{}\' not found in camera calibration data'.format(
                    camera_id
                ))
            camera_calibration = camera_calibrations.get(camera_id)
    if pose_label_column is not None:
        if pose_label_column not in poses.columns:
            raise ValueError('Specified pose label column \'{}\' not found in pose data'.format(
                pose_label_column
            ))
        if pose_label_map is not None:
            pose_label_map = poseconnect.utils.convert_to_lookup_dict(pose_label_map)
        elif generate_pose_label_map:
            source_labels = (
                poses
                .sort_values('timestamp')
                .loc[:, pose_label_column]
                .dropna()
                .drop_duplicates()
                .tolist()
            )
            target_labels = list(range(len(source_labels)))
            pose_label_map = dict(zip(source_labels, target_labels))
        if pose_label_map is not None:
            poses[pose_label_column] = poses[pose_label_column].apply(
                lambda x: pose_label_map.get(x)
            )
    timestamp_text_color = matplotlib.colors.to_hex(timestamp_text_color, keep_alpha=False)
    timestamp_box_color = matplotlib.colors.to_hex(timestamp_box_color, keep_alpha=False)
    if pose_type == '2d' and 'camera_id' in poses.columns:
        if camera_id is not None:
            poses = poses.loc[poses['camera_id'] == camera_id].copy()
        if poses['camera_id'].nunique() > 1:
            raise ValueError('Pose data contains multiple camera IDs for a single video')
    elif pose_type == '3d':
        if camera_calibration is None:
            raise ValueError('Camera calibration parameters must be specified to overlay 3D poses')
    else:
        raise ValueError('Pose type must be either \'2d\' or \'3d\'')
    logger.info('Ingested {} poses spanning time period {} to {}'.format(
        len(poses),
        poses['timestamp'].min().isoformat(),
        poses['timestamp'].max().isoformat()
    ))
    logger.info('Video input path: {}'.format(video_input_path))
    logger.info('Video start time: {}'.format(
        video_start_time.isoformat()
    ))
    video_input = cv_utils.VideoInput(
        input_path=video_input_path,
        start_time=video_start_time
    )
    if video_fps is None:
        video_fps = video_input.video_parameters.fps
        if video_fps is None:
            raise ValueError('Failed to rate video frame rate from video file.')
    logger.info('Video frame rate: {} frames per second'.format(
        video_fps
    ))
    if video_frame_count is None:
        video_frame_count = video_input.video_parameters.frame_count
        if video_frame_count is None:
            raise ValueError('Failed to rate video frame count from video file.')
    logger.info('Video frame count: {}'.format(
        video_frame_count
    ))
    logger.info('Video input path: {}'.format(video_input_path))
    if image_output_path is None:
        video_input_directory, video_input_filename = os.path.split(video_input_path)
        video_input_filename_stem, video_input_filename_extension = os.path.splitext(video_input_filename)
        logger.info('Video input directory: {}'.format(video_input_directory))
        logger.info('Video input filename stem: {}'.format(video_input_filename_stem))
        if image_output_directory is None:
            image_output_directory = video_input_directory
        image_output_filename_stem = '_'.join([
            video_input_filename_stem,
            image_output_filename_suffix,
            timestamp.strftime('%Y%m%d_%H%M%S_%f')
        ])
        logger.info('Image output directory: {}'.format(image_output_directory))
        logger.info('Image output filename stem: {}'.format(image_output_filename_stem))
        logger.info('Image output filename extension: {}'.format(image_output_filename_extension))
        image_output_path = os.path.join(
            image_output_directory,
            '.'.join([
                image_output_filename_stem,
                image_output_filename_extension
            ])
        )
    logger.info('Image output path: {}'.format(image_output_path))
    nearest_pose_timestamp = find_nearest_pose_timestamp(
        pose_timestamps=poses['timestamp'],
        target_timestamp=timestamp,
        timestamp_offset_max_milliseconds=timestamp_offset_max_milliseconds
    )
    if nearest_pose_timestamp is None:
        logger.info('No pose timestamp near target timestamp')
        return
    frame = video_input.get_frame_by_timestamp(timestamp)
    video_input.close()
    if frame is None:
        raise ValueError('No video frame at target timestamp {}'.format(timestamp.isoformat()))
    poses_frame = poses.loc[poses['timestamp'] == nearest_pose_timestamp].copy()
    logger.info('Found {} poses for chosen frame'.format(
        len(poses_frame)
    ))
    frame=overlay_poses_image(
        poses=poses_frame,
        image=frame,
        pose_type=pose_type,
        camera_calibration=camera_calibration,
        pose_label_column=pose_label_column,
        draw_keypoint_connectors=draw_keypoint_connectors,
        keypoint_connectors=keypoint_connectors,
        pose_model_name=pose_model_name,
        pose_color=pose_color,
        keypoint_radius=keypoint_radius,
        keypoint_alpha=keypoint_alpha,
        keypoint_connector_alpha=keypoint_connector_alpha,
        keypoint_connector_linewidth=keypoint_connector_linewidth,
        pose_label_text_color=pose_label_text_color,
        pose_label_box_alpha=pose_label_box_alpha,
        pose_label_font_scale=pose_label_font_scale,
        pose_label_text_line_width=pose_label_text_line_width
    )
    if draw_timestamp:
        frame = cv_utils.draw_timestamp(
            original_image=frame,
            timestamp=timestamp,
            padding=timestamp_padding,
            font_scale=timestamp_font_scale,
            text_line_width=timestamp_text_line_width,
            text_color=timestamp_text_color,
            text_alpha=1.0,
            box_line_width=0,
            box_color=timestamp_box_color,
            box_fill=True,
            box_alpha=timestamp_box_alpha
        )
    cv_utils.write_image(
        image=frame,
        path=image_output_path
    )

def overlay_poses_image(
    poses,
    image,
    pose_type=poseconnect.defaults.OVERLAY_POSE_TYPE,
    camera_calibration=poseconnect.defaults.OVERLAY_CAMERA_CALIBRATION,
    pose_label_column=poseconnect.defaults.OVERLAY_POSE_LABEL_COLUMN,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_text_color=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_COLOR,
    pose_label_box_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BOX_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_text_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_LINE_WIDTH,
):
    if pose_type == '2d':
        if poses['camera_id'].nunique() > 1:
            raise ValueError('Pose data contains multiple camera IDs for a single image')
        keypoint_coordinate_column_name='keypoint_coordinates_2d'
    elif pose_type == '3d':
        if camera_calibration is None:
            raise ValueError('Camera calibration parameters must be specified to overlay 3D poses')
        keypoint_coordinate_column_name='keypoint_coordinates_3d'
    else:
        raise ValueError('Pose type must be either \'2d\' or \'3d\'')
    if poses['timestamp'].nunique() > 1:
        raise ValueError('Pose data contains multiple timestamps for a single image')
    for pose_id, row in poses.iterrows():
        image = overlay_pose_image(
            keypoint_coordinates=row.get(keypoint_coordinate_column_name),
            image=image,
            pose_type=pose_type,
            camera_calibration=camera_calibration,
            pose_label=row.get(pose_label_column),
            draw_keypoint_connectors=draw_keypoint_connectors,
            keypoint_connectors=keypoint_connectors,
            pose_model_name=pose_model_name,
            pose_color=pose_color,
            keypoint_radius=keypoint_radius,
            keypoint_alpha=keypoint_alpha,
            keypoint_connector_alpha=keypoint_connector_alpha,
            keypoint_connector_linewidth=keypoint_connector_linewidth,
            pose_label_text_color=pose_label_text_color,
            pose_label_box_alpha=pose_label_box_alpha,
            pose_label_font_scale=pose_label_font_scale,
            pose_label_text_line_width=pose_label_text_line_width
        )
    return image

def overlay_pose_image(
    keypoint_coordinates,
    image,
    pose_type=poseconnect.defaults.OVERLAY_POSE_TYPE,
    camera_calibration=poseconnect.defaults.OVERLAY_CAMERA_CALIBRATION,
    pose_label=None,
    draw_keypoint_connectors=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    keypoint_connectors=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTORS,
    pose_model_name=poseconnect.defaults.OVERLAY_POSE_MODEL_NAME,
    pose_color=poseconnect.defaults.OVERLAY_POSE_COLOR,
    keypoint_radius=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    keypoint_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    keypoint_connector_alpha=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    keypoint_connector_linewidth=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    pose_label_text_color=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_COLOR,
    pose_label_box_alpha=poseconnect.defaults.OVERLAY_POSE_LABEL_BOX_ALPHA,
    pose_label_font_scale=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    pose_label_text_line_width=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_LINE_WIDTH
):
    keypoint_coordinates = np.asarray(keypoint_coordinates)
    if pose_type == '2d':
        if keypoint_coordinates.shape[-1] != 2:
            raise ValueError('Expected 2D pose but keypoint coordinates have shape {}'.format(
                keypoint_coordinates.shape
            ))
    elif pose_type == '3d':
        if keypoint_coordinates.shape[-1] != 3:
            raise ValueError('Expected 3D pose but keypoint coordinates have shape {}'.format(
                keypoint_coordinates.shape
            ))
        if camera_calibration is None:
            raise ValueError('Camera calibration parameters must be specified to overlay 3D poses')
        keypoint_coordinates = cv_utils.project_points(
            keypoint_coordinates,
            rotation_vector=camera_calibration['rotation_vector'],
            translation_vector=camera_calibration['translation_vector'],
            camera_matrix=camera_calibration['camera_matrix'],
            distortion_coefficients=camera_calibration['distortion_coefficients'],
            remove_behind_camera=True,
            remove_outside_frame=True,
            image_corners=[
                [0,0],
                [camera_calibration['image_width'], camera_calibration['image_height']]
            ]
        )
    else:
        raise ValueError('Pose type must be either \'2d\' or \'3d\'')
    pose_color = matplotlib.colors.to_hex(pose_color, keep_alpha=False)
    pose_label_text_color = matplotlib.colors.to_hex(pose_label_text_color, keep_alpha=False)
    if draw_keypoint_connectors:
        if keypoint_connectors is None:
            if pose_model_name is None:
                raise ValueError('Must specify pose model name or explicitly specify keypoint connectors to draw keypoint connectors')
            if pose_model_name not in KEYPOINT_CONNECTORS_BY_POSE_MODEL.keys():
                raise ValueError('Pose model \'{}\' not recognized'.format(
                    pose_model_name
                ))
            keypoint_connectors = KEYPOINT_CONNECTORS_BY_POSE_MODEL[pose_model_name]
    if not np.any(np.all(np.isfinite(keypoint_coordinates), axis=1), axis=0):
        return image
    valid_keypoints = np.all(np.isfinite(keypoint_coordinates), axis=1)
    plottable_points = keypoint_coordinates[valid_keypoints]
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
                        keypoint_coordinates[keypoint_from_index],
                        keypoint_coordinates[keypoint_to_index]
                    ],
                    line_width=keypoint_connector_linewidth,
                    color=pose_color,
                    alpha=keypoint_connector_alpha
                )
    if pd.notna(pose_label):
        pose_label_anchor = np.nanmean(keypoint_coordinates, axis=0)
        new_image = cv_utils.draw_text_box(
            original_image=new_image,
            anchor_coordinates=pose_label_anchor,
            text=str(pose_label),
            horizontal_alignment='center',
            vertical_alignment='middle',
            font_scale=pose_label_font_scale,
            text_line_width=pose_label_text_line_width,
            text_color=pose_label_text_color,
            text_alpha=1.0,
            box_line_width=1.5,
            box_color=pose_color,
            box_fill=True,
            box_alpha=pose_label_box_alpha
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
            logger.debug('There are no pose timestamps nearby video timestamp {}.'.format(
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
            logger.debug('There are {} pose timestamps nearby video timestamp {}: {}. Chose pose timestamp {}'.format(
                len(nearby_pose_timestamps),
                video_timestamp.isoformat(),
                [nearby_pose_timestamp.isoformat() for nearby_pose_timestamp in nearby_pose_timestamps],
                aligned_pose_timestamp.isoformat()
            ))
        aligned_pose_timestamps.append(aligned_pose_timestamp)
    aligned_pose_timestamps = pd.DatetimeIndex(aligned_pose_timestamps)
    return video_timestamps, aligned_pose_timestamps

def find_nearest_pose_timestamp(
    pose_timestamps,
    target_timestamp,
    timestamp_offset_max_milliseconds=poseconnect.defaults.OVERLAY_TIMESTAMP_OFFSET_MAX_MILLISECONDS
):
    pose_timestamps = pd.DatetimeIndex(
        pd.to_datetime(pose_timestamps, utc=True)
        .drop_duplicates()
        .sort_values()
    )
    target_timestamp = poseconnect.utils.convert_to_datetime_utc(target_timestamp)
    nearby_pose_timestamps = pose_timestamps[
        (pose_timestamps >= target_timestamp - datetime.timedelta(milliseconds=timestamp_offset_max_milliseconds)) &
        (pose_timestamps < target_timestamp + datetime.timedelta(milliseconds=timestamp_offset_max_milliseconds))
    ]
    if len(nearby_pose_timestamps) == 0:
        logger.debug('There are no pose timestamps nearby target timestamp {}.'.format(
            target_timestamp.isoformat()
        ))
        nearest_pose_timestamp = None
    elif len(nearby_pose_timestamps) == 1:
        nearest_pose_timestamp = nearby_pose_timestamps[0]
    else:
        time_distances = [
            max(nearby_pose_timestamp.to_pydatetime(), target_timestamp) -
            min(nearby_pose_timestamp.to_pydatetime(), target_timestamp)
            for nearby_pose_timestamp in nearby_pose_timestamps
        ]
        nearest_pose_timestamp = nearby_pose_timestamps[np.argmin(time_distances)]
        logger.debug('There are {} pose timestamps nearby video timestamp {}: {}. Chose pose timestamp {}'.format(
            len(nearby_pose_timestamps),
            target_timestamp.isoformat(),
            [nearby_pose_timestamp.isoformat() for nearby_pose_timestamp in nearby_pose_timestamps],
            nearest_pose_timestamp.isoformat()
        ))
    return nearest_pose_timestamp
