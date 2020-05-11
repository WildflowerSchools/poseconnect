import process_pose_data.visualize
import process_pose_data.fetch
import video_io
import pandas as pd
import numpy as np
import cv_utils
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import tqdm
import slugify
import datetime
import string
import logging
import os

logger = logging.getLogger(__name__)

def overlay_video_poses_3d(
    poses_3d_df,
    video_start,
    camera_ids,
    pose_model_id=None,
    camera_names=None,
    camera_calibrations=None,
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    pose_label=None,
    pose_color='green',
    keypoint_radius=3,
    keypoint_alpha=0.6,
    keypoint_connector_alpha=0.6,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.6,
    pose_label_font_scale=2.0,
    pose_label_line_width=2,
    output_directory='.',
    output_filename_prefix='poses_3d_overlay',
    output_filename_datetime_format='%Y%m%d_%H%M%S_%f',
    output_filename_extension='mp4',
    progress_bar=False,
    notebook=False
):
    if pose_model_id is not None:
        pose_model = process_pose_data.fetch.fetch_pose_model_by_pose_model_id(
            pose_model_id
        )
        if keypoint_connectors is None:
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if camera_names is None:
        camera_names = process_pose_data.fetch.fetch_camera_names(
            camera_ids
        )
    if camera_calibrations is None:
        camera_calibrations = process_pose_data.fetch.fetch_camera_calibrations(
            camera_ids,
            start=timestamp.to_pydatetime(),
            end=timestamp.to_pydatetime()
        )
    for camera_id in camera_ids:
        camera_name = camera_names[camera_id]
        camera_calibration = camera_calibrations[camera_id]
        logger.info('Overlaying poses for {}'.format(camera_name))
        video_metadata_with_local_paths = video_io.fetch_videos(
            video_timestamps=[video_start],
            camera_device_ids=[camera_id],
        )
        if len(video_metadata_with_local_paths) > 1:
            raise ValueError('More than one video found for camera ID {} and video start {}'.format(
                camera_id,
                video_start.isoformat()
            ))
        input_path = video_metadata_with_local_paths[0]['video_local_path']
        logger.info('Video input path: {}'.format(input_path))
        output_path = os.path.join(
            output_directory,
            '{}_{}_{}.{}'.format(
                output_filename_prefix,
                slugify.slugify(camera_name),
                video_start.strftime(output_filename_datetime_format),
                output_filename_extension
            )
        )
        logger.info('Video output path: {}'.format(output_path))
        if keypoint_connectors is not None:
            draw_keypoint_connectors = True
        else:
            draw_keypoint_connectors = False
        video_input = cv_utils.VideoInput(
            input_path=input_path,
            start_time=video_start
        )
        video_start_time = video_input.video_parameters.start_time
        video_fps = video_input.video_parameters.fps
        video_frame_count = video_input.video_parameters.frame_count
        logger.info('Opened video input. Start time: {}. FPS: {}. Frame count: {}'.format(
            video_start_time.isoformat(),
            video_fps,
            video_frame_count
        ))
        if video_fps != 10.0:
            raise ValueError('Overlay function expects 10 FPS but video has {} FPS'.format(video_fps))
        if pd.to_datetime(video_start_time, utc=True) < poses_3d_df['timestamp'].min():
            raise ValueError('Video starts at {} but 3D pose data starts at {}'.format(
                video_start_time.isoformat(),
                poses_3d_df['timestamp'].min().isoformat()
            ))
        video_end_time = video_start_time + datetime.timedelta(milliseconds=(video_frame_count - 1)*100)
        if pd.to_datetime(video_end_time, utc=True) > poses_3d_df['timestamp'].max():
            raise ValueError('Video ends at {} but 3D pose data ends at {}'.format(
                video_end_time.isoformat(),
                poses_3d_df['timestamp'].max().isoformat()
            ))
        video_output = cv_utils.VideoOutput(
            output_path,
            video_parameters=video_input.video_parameters
        )
        if progress_bar:
            if notebook:
                t = tqdm.tqdm_notebook(total=video_frame_count)
            else:
                t = tqdm.tqdm(total=video_frame_count)
        for frame_index in range(video_frame_count):
            timestamp = video_start + datetime.timedelta(milliseconds=frame_index*100)
            timestamp_pandas = pd.to_datetime(timestamp, utc=True)
            frame = video_input.get_frame()
            if frame is None:
                raise ValueError('Input video ended unexpectedly at frame number {}'.format(frame_index))
            for pose_3d_id, row in poses_3d_df.loc[poses_3d_df['timestamp'] == timestamp_pandas].iterrows():
                keypoint_coordinates_2d = cv_utils.project_points(
                    object_points=row['keypoint_coordinates_3d'],
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
                frame=draw_pose_2d_opencv(
                    image=frame,
                    keypoint_coordinates=keypoint_coordinates_2d,
                    draw_keypoint_connectors=draw_keypoint_connectors,
                    keypoint_connectors=keypoint_connectors,
                    keypoint_alpha=keypoint_alpha,
                    keypoint_connector_alpha=keypoint_connector_alpha,
                    keypoint_connector_linewidth=keypoint_connector_linewidth,
                )
            video_output.write_frame(frame)
            if progress_bar:
                t.update()
        video_input.close()
        video_output.close()

def draw_poses_2d_timestamp_camera_pair_opencv(
    df,
    annotate_matches=False,
    generate_match_aliases=False,
    camera_names={'a': None, 'b': None},
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    keypoint_alpha=0.6,
    keypoint_connector_alpha=0.6,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.6,
    pose_label_font_scale=2.0,
    pose_label_line_width=2,
    show=True,
    fig_width_inches=10.5,
    fig_height_inches=8
):
    dfs_single_camera = process_pose_data.visualize.extract_single_camera_data(df)
    pose_label_maps = {
        'a': None,
        'b': None
    }
    pose_color_maps = {
        'a': None,
        'b': None
    }
    if annotate_matches:
        num_matches = df['match'].sum()
        for camera_letter in ['a', 'b']:
            pose_ids = dfs_single_camera[camera_letter].index.values.tolist()
            pose_label_maps[camera_letter] = {pose_id: '' for pose_id in pose_ids}
            if not generate_match_aliases:
                if 'track_label' in dfs_single_camera[camera_letter].columns:
                    pose_labels = dfs_single_camera[camera_letter]['track_label'].values.tolist()
                elif len(pose_ids) <= 26:
                    pose_labels = string.ascii_uppercase[:len(pose_ids)]
                else:
                    pose_labels = range(len(pose_ids))
                pose_label_maps[camera_letter] = dict(zip(pose_ids, pose_labels))
            pose_color_maps[camera_letter] = {pose_id: 'grey' for pose_id in pose_ids}
        match_aliases = iter(list(string.ascii_uppercase[:num_matches]))
        match_colors = iter(sns.color_palette('husl', n_colors=num_matches))
        for (pose_id_a, pose_id_b), row in df.iterrows():
            if row['match']:
                old_label_a = pose_label_maps['a'][pose_id_a]
                old_label_b = pose_label_maps['b'][pose_id_b]
                pose_label_maps['a'][pose_id_a] = '{} ({})'.format(
                    old_label_a,
                    old_label_b
                )
                pose_label_maps['b'][pose_id_b] = '{} ({})'.format(
                    old_label_b,
                    old_label_a
                )
                if generate_match_aliases:
                    match_alias = next(match_aliases)
                    pose_label_maps['a'][pose_id_a] = match_alias
                    pose_label_maps['b'][pose_id_b] = match_alias
                pose_color = next(match_colors)
                pose_color_maps['a'][pose_id_a] = pose_color
                pose_color_maps['b'][pose_id_b] = pose_color
    for camera_letter in ['a', 'b']:
        draw_poses_2d_timestamp_camera_opencv(
            df=dfs_single_camera[camera_letter],
            draw_keypoint_connectors=draw_keypoint_connectors,
            keypoint_connectors=keypoint_connectors,
            pose_label_map=pose_label_maps[camera_letter],
            pose_color_map=pose_color_maps[camera_letter],
            keypoint_alpha=keypoint_alpha,
            keypoint_connector_alpha=keypoint_connector_alpha,
            keypoint_connector_linewidth=keypoint_connector_linewidth,
            pose_label_color=pose_label_color,
            pose_label_background_alpha=pose_label_background_alpha,
            pose_label_font_scale=pose_label_font_scale,
            show=show,
            fig_width_inches=fig_width_inches,
            fig_height_inches=fig_height_inches
        )

def draw_poses_2d_timestamp_camera_opencv(
    df,
    background_image=None,
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    pose_label_map=None,
    pose_color_map=None,
    keypoint_alpha=0.6,
    keypoint_connector_alpha=0.6,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.6,
    pose_label_font_scale=2.0,
    pose_label_line_width=2,
    show=True,
    fig_width_inches=10.5,
    fig_height_inches=8
):
    timestamps = df['timestamp'].unique()
    camera_ids = df['camera_id'].unique()
    if len(timestamps) > 1:
        raise ValueError('More than one timestamp in data frame')
    if len(camera_ids) > 1:
        raise ValueError('More than one camera in data frame')
    timestamp = timestamps[0]
    camera_id = camera_ids[0]
    camera_identifier = camera_id
    df = df.sort_index()
    pose_ids = df.index.tolist()
    if pose_label_map is None:
        if 'track_label' in df.columns:
            pose_labels = df['track_label'].values.tolist()
        elif len(pose_ids) <= 26:
            pose_labels = string.ascii_uppercase[:len(pose_ids)]
        else:
            pose_labels = range(len(pose_ids))
        pose_label_map = dict(zip(pose_ids, pose_labels))
    if pose_color_map is None:
        pose_colors = sns.color_palette('husl', n_colors=len(pose_ids))
        pose_color_map = dict(zip(pose_ids, pose_colors))
    if draw_keypoint_connectors:
        if keypoint_connectors is None:
            pose_model = process_pose_data.fetch.fetch_pose_model(
                pose_id=pose_ids[0]
            )
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if background_image is None:
        image_metadata = video_io.fetch_images(
            image_timestamps = [timestamp.to_pydatetime()],
            camera_device_ids=[camera_id]
        )
        image_local_path = image_metadata[0]['image_local_path']
        background_image = cv_utils.fetch_image_from_local_drive(image_local_path)
    new_image = background_image
    for pose_id, row in df.iterrows():
        new_image = process_pose_data.draw_pose_2d_opencv(
            image=new_image,
            keypoint_coordinates=row['keypoint_coordinates'],
            draw_keypoint_connectors=draw_keypoint_connectors,
            keypoint_connectors=keypoint_connectors,
            pose_label=pose_label_map[pose_id],
            pose_color=pose_color_map[pose_id],
            keypoint_alpha=keypoint_alpha,
            keypoint_connector_alpha=keypoint_connector_alpha,
            keypoint_connector_linewidth=keypoint_connector_linewidth,
            pose_label_color=pose_label_color,
            pose_label_background_alpha=pose_label_background_alpha,
            pose_label_font_scale=pose_label_font_scale,
            pose_label_line_width=pose_label_line_width
        )
    # Show plot
    if show:
        fig, ax = plt.subplots()
        ax.imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))
        ax.axis('off')
        fig.set_size_inches(fig_width_inches, fig_height_inches)
    return new_image

def draw_pose_2d_opencv(
    image,
    keypoint_coordinates,
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    pose_label=None,
    pose_color='green',
    keypoint_radius=3,
    keypoint_alpha=0.6,
    keypoint_connector_alpha=0.6,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.6,
    pose_label_font_scale=2.0,
    pose_label_line_width=2
):
    pose_color = matplotlib.colors.to_hex(pose_color, keep_alpha=False)
    pose_label_color = matplotlib.colors.to_hex(pose_label_color, keep_alpha=False)
    keypoint_coordinates = np.asarray(keypoint_coordinates).reshape((-1, 2))
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
    if pose_label is not None:
        pose_label_anchor = np.nanmean(keypoint_coordinates, axis=0)
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

def draw_poses_3d_timestamp_camera_opencv(
    df,
    camera_ids,
    pose_model_id=None,
    camera_names=None,
    camera_calibrations=None,
    keypoint_connectors=None,
    keypoint_alpha=0.6,
    keypoint_connector_alpha=0.6,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.6,
    show=True,
    fig_width_inches=10.5,
    fig_height_inches=8
):
    timestamps = df['timestamp'].unique()
    if len(timestamps) > 1:
        raise ValueError('More than one timestamp in data frame')
    timestamp = timestamps[0]
    if pose_model_id is not None:
        pose_model = process_pose_data.fetch.fetch_pose_model_by_pose_model_id(
            pose_model_id
        )
        if keypoint_connectors is None:
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if camera_names is None:
        camera_names = process_pose_data.fetch.fetch_camera_names(
            camera_ids
        )
    if camera_calibrations is None:
        camera_calibrations = process_pose_data.fetch.fetch_camera_calibrations(
            camera_ids,
            start=timestamp.to_pydatetime(),
            end=timestamp.to_pydatetime()
        )
    match_group_labels = df['match_group_label'].unique()
    color_mapping = process_pose_data.visualize.generate_color_mapping(match_group_labels)
    for camera_id in camera_ids:
        camera_name = camera_names[camera_id]
        camera_calibration = camera_calibrations[camera_id]
        if keypoint_connectors is not None:
            draw_keypoint_connectors = True
        else:
            draw_keypoint_connectors = False
        image_metadata = video_io.fetch_images(
            image_timestamps = [timestamp.to_pydatetime()],
            camera_device_ids=[camera_id]
        )
        image_local_path = image_metadata[0]['image_local_path']
        background_image = cv_utils.fetch_image_from_local_drive(image_local_path)
        new_image = background_image
        for pose_3d_id, row in df.iterrows():
            keypoint_coordinates_2d = cv_utils.project_points(
                object_points=row['keypoint_coordinates_3d'],
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
            new_image=draw_pose_2d_opencv(
                image=new_image,
                keypoint_coordinates=keypoint_coordinates_2d,
                draw_keypoint_connectors=draw_keypoint_connectors,
                keypoint_connectors=keypoint_connectors,
                pose_label=row['match_group_label'],
                pose_color=color_mapping[row['match_group_label']],
                keypoint_alpha=keypoint_alpha,
                keypoint_connector_alpha=keypoint_connector_alpha,
                keypoint_connector_linewidth=keypoint_connector_linewidth,
                pose_label_color=pose_label_color,
                pose_label_background_alpha=pose_label_background_alpha
            )
            # Show plot
        if show:
            fig, ax = plt.subplots()
            ax.imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))
            ax.axis('off')
            fig.set_size_inches(fig_width_inches, fig_height_inches)
