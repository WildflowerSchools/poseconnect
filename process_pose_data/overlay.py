import process_pose_data.visualize
import process_pose_data.honeycomb_io
import video_io
import pandas as pd
import numpy as np
import cv_utils
import cv2 as cv
import ffmpeg
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

def overlay_video_poses_2d(
    poses_2d_df,
    video_start,
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
    output_filename_prefix='poses_2d_overlay',
    output_filename_datetime_format='%Y%m%d_%H%M%S_%f',
    output_filename_extension='mp4',
    output_fourcc_string=None,
    progress_bar=False,
    notebook=False
):
    camera_ids = poses_2d_df['camera_id'].unique()
    if pose_model_id is not None:
        pose_model = process_pose_data.honeycomb_io.fetch_pose_model_by_pose_model_id(
            pose_model_id
        )
        if keypoint_connectors is None:
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if camera_names is None:
        camera_names = process_pose_data.honeycomb_io.fetch_camera_names(
            camera_ids
        )
    if camera_calibrations is None:
        camera_calibrations = process_pose_data.honeycomb_io.fetch_camera_calibrations(
            camera_ids,
            start=video_start,
            end=video_start
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
                video_start.strftime(output_filename_datetime_format),
                slugify.slugify(camera_name),
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
        if pd.to_datetime(video_start_time, utc=True) < poses_2d_df['timestamp'].min():
            raise ValueError('Video starts at {} but 2D pose data starts at {}'.format(
                video_start_time.isoformat(),
                poses_2d_df['timestamp'].min().isoformat()
            ))
        video_end_time = video_start_time + datetime.timedelta(milliseconds=(video_frame_count - 1)*100)
        if pd.to_datetime(video_end_time, utc=True) > poses_2d_df['timestamp'].max():
            raise ValueError('Video ends at {} but 2D pose data ends at {}'.format(
                video_end_time.isoformat(),
                poses_2d_df['timestamp'].max().isoformat()
            ))
        video_output_parameters = video_input.video_parameters
        if output_fourcc_string is not None:
            video_output_parameters.fourcc_int = cv_utils.fourcc_string_to_int(output_fourcc_string)
        video_output = cv_utils.VideoOutput(
            output_path,
            video_parameters=video_output_parameters
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
            for pose_id_2d, row in poses_2d_df.loc[
                (poses_2d_df['timestamp'] == timestamp_pandas) &
                (poses_2d_df['camera_id'] == camera_id)
            ].iterrows():
                keypoint_coordinates_2d = row['keypoint_coordinates_2d']
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

def overlay_video_poses_3d(
    poses_3d_df,
    start=None,
    end=None,
    video_timestamps=None,
    camera_assignment_ids=None,
    environment_id=None,
    environment_name=None,
    camera_device_types=video_io.DEFAULT_CAMERA_DEVICE_TYPES,
    camera_device_ids=None,
    camera_part_numbers=None,
    camera_names=None,
    camera_serial_numbers=None,
    chunk_size=100,
    minimal_honeycomb_client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    local_video_directory='./videos',
    video_filename_extension='mp4',
    pose_model_id=None,
    camera_calibrations=None,
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    pose_label_column=None,
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
    output_fourcc_string=None,
    concatenate_videos=True,
    delete_individual_clips=True,
    progress_bar=False,
    notebook=False
):
    video_metadata_with_local_paths = video_io.fetch_videos(
        start=start,
        end=end,
        video_timestamps=video_timestamps,
        camera_assignment_ids=camera_assignment_ids,
        environment_id=environment_id,
        environment_name=environment_name,
        camera_device_types=camera_device_types,
        camera_device_ids=camera_device_ids,
        camera_part_numbers=camera_part_numbers,
        camera_names=camera_names,
        camera_serial_numbers=camera_serial_numbers,
        chunk_size=chunk_size,
        minimal_honeycomb_client=minimal_honeycomb_client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret,
        local_video_directory=local_video_directory,
        video_filename_extension=video_filename_extension
    )
    video_metadata_dict = dict()
    video_timestamp_min = None
    video_timestamp_max = None
    for datum in video_metadata_with_local_paths:
        camera_id = datum.get('device_id')
        video_timestamp = datum.get('video_timestamp')
        if camera_id not in video_metadata_dict.keys():
            video_metadata_dict[camera_id] = dict()
        video_metadata_dict[camera_id][video_timestamp] = datum
        if video_timestamp_min is None or video_timestamp < video_timestamp_min:
            video_timestamp_min = video_timestamp
        if video_timestamp_max is None or video_timestamp > video_timestamp_max:
            video_timestamp_max = video_timestamp
    camera_ids = list(video_metadata_dict.keys())
    camera_name_dict = process_pose_data.honeycomb_io.fetch_camera_names(
        camera_ids
    )
    if camera_calibrations is None:
        camera_calibrations = process_pose_data.honeycomb_io.fetch_camera_calibrations(
            camera_ids,
            start=video_timestamp_min,
            end=video_timestamp_max
        )
    if pose_model_id is not None:
        pose_model = process_pose_data.honeycomb_io.fetch_pose_model_by_pose_model_id(
            pose_model_id
        )
        if keypoint_connectors is None:
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if keypoint_connectors is not None:
        draw_keypoint_connectors = True
    else:
        draw_keypoint_connectors = False
    for camera_id in camera_ids:
        camera_name = camera_name_dict[camera_id]
        camera_calibration = camera_calibrations[camera_id]
        logger.info('Overlaying poses for {}'.format(camera_name))
        video_timestamps = sorted(video_metadata_dict[camera_id].keys())
        output_paths = list()
        for video_timestamp in video_timestamps:
            logger.info('Overlaying poses for video starting at {}'.format(video_timestamp.isoformat()))
            input_path = video_metadata_dict[camera_id][video_timestamp]['video_local_path']
            logger.info('Video input path: {}'.format(input_path))
            output_path = os.path.join(
                output_directory,
                '{}_{}_{}.{}'.format(
                    output_filename_prefix,
                    video_timestamp.strftime(output_filename_datetime_format),
                    slugify.slugify(camera_name),
                    output_filename_extension
                )
            )
            output_paths.append(output_path)
            logger.info('Video output path: {}'.format(output_path))
            video_input = cv_utils.VideoInput(
                input_path=input_path,
                start_time=video_timestamp
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
            video_output_parameters = video_input.video_parameters
            if output_fourcc_string is not None:
                video_output_parameters.fourcc_int = cv_utils.fourcc_string_to_int(output_fourcc_string)
            video_output = cv_utils.VideoOutput(
                output_path,
                video_parameters=video_output_parameters
            )
            if progress_bar:
                if notebook:
                    t = tqdm.tqdm_notebook(total=video_frame_count)
                else:
                    t = tqdm.tqdm(total=video_frame_count)
            for frame_index in range(video_frame_count):
                timestamp = video_timestamp + datetime.timedelta(milliseconds=frame_index*100)
                timestamp_pandas = pd.to_datetime(timestamp, utc=True)
                frame = video_input.get_frame()
                if frame is None:
                    raise ValueError('Input video ended unexpectedly at frame number {}'.format(frame_index))
                for pose_id_3d, row in poses_3d_df.loc[poses_3d_df['timestamp'] == timestamp_pandas].iterrows():
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
                    if pose_label_column is not None:
                        pose_label = row[pose_label_column]
                    else:
                        pose_label = None
                    frame=draw_pose_2d_opencv(
                        image=frame,
                        keypoint_coordinates=keypoint_coordinates_2d,
                        pose_label=pose_label,
                        draw_keypoint_connectors=draw_keypoint_connectors,
                        keypoint_connectors=keypoint_connectors,
                        keypoint_alpha=keypoint_alpha,
                        keypoint_connector_alpha=keypoint_connector_alpha,
                        keypoint_connector_linewidth=keypoint_connector_linewidth,
                        pose_label_font_scale=pose_label_font_scale,
                        pose_label_line_width=pose_label_line_width
                    )
                video_output.write_frame(frame)
                if progress_bar:
                    t.update()
            video_input.close()
            video_output.close()
        if concatenate_videos:
            concat_output_path = os.path.join(
                output_directory,
                '{}_{}_{}_{}.{}'.format(
                    output_filename_prefix,
                    video_timestamps[0].strftime(output_filename_datetime_format),
                    video_timestamps[-1].strftime(output_filename_datetime_format),
                    slugify.slugify(camera_name),
                    output_filename_extension
                )
            )
            logger.info('Concatenating videos from {} to {} into {}'.format(
                output_paths[0],
                output_paths[-1],
                concat_output_path
            ))
            concat_videos(
                input_videos_path_list=output_paths,
                output_video_path=concat_output_path,
                delete_input_videos=delete_individual_clips
            )



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
                if 'track_label_2d' in dfs_single_camera[camera_letter].columns:
                    pose_labels = dfs_single_camera[camera_letter]['track_label_2d'].values.tolist()
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
        if 'track_label_2d' in df.columns:
            pose_labels = df['track_label_2d'].values.tolist()
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
            pose_model = process_pose_data.honeycomb_io.fetch_pose_model(
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
            keypoint_coordinates=row['keypoint_coordinates_2d'],
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
        pose_model = process_pose_data.honeycomb_io.fetch_pose_model_by_pose_model_id(
            pose_model_id
        )
        if keypoint_connectors is None:
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if camera_names is None:
        camera_names = process_pose_data.honeycomb_io.fetch_camera_names(
            camera_ids
        )
    if camera_calibrations is None:
        camera_calibrations = process_pose_data.honeycomb_io.fetch_camera_calibrations(
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
        for pose_id_3d, row in df.iterrows():
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

def draw_poses_2d_timestamp_camera_pair(
    df,
    annotate_matches=False,
    generate_match_aliases=False,
    camera_names={'a': None, 'b': None},
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    keypoint_alpha=0.3,
    keypoint_connector_alpha=0.3,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.5,
    pose_label_boxstyle='circle',
    image_size=None,
    show_axes=False,
    show_background_image=True,
    background_image=None,
    background_image_alpha=0.4,
    display_camera_name=False,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S.%f',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='poses_2d',
    filename_datetime_format='%Y%m%d_%H%M%S_%f',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    dfs_single_camera = extract_single_camera_data(df)
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
                if 'track_label_2d' in dfs_single_camera[camera_letter].columns:
                    pose_labels = dfs_single_camera[camera_letter]['track_label_2d'].values.tolist()
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
        draw_poses_2d_timestamp_camera(
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
            pose_label_boxstyle=pose_label_boxstyle,
            image_size=image_size,
            show_axes=show_axes,
            show_background_image=show_background_image,
            background_image=background_image,
            background_image_alpha=background_image_alpha,
            display_camera_name=display_camera_name,
            camera_name=camera_names[camera_letter],
            plot_title_datetime_format=plot_title_datetime_format,
            show=show,
            save=save,
            save_directory=save_directory,
            filename_prefix=filename_prefix,
            filename_datetime_format=filename_datetime_format,
            filename_extension=filename_extension,
            fig_width_inches=fig_width_inches,
            fig_height_inches=fig_height_inches
        )

def draw_poses_2d_timestamp_camera(
    df,
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    pose_label_map=None,
    pose_color_map=None,
    keypoint_alpha=0.3,
    keypoint_connector_alpha=0.3,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.5,
    pose_label_boxstyle='circle',
    image_size=None,
    show_axes=False,
    show_background_image=True,
    background_image=None,
    background_image_alpha=0.4,
    display_camera_name=False,
    camera_name=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S.%f',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='poses_2d',
    filename_datetime_format='%Y%m%d_%H%M%S_%f',
    filename_extension='png',
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
    if display_camera_name:
        if camera_name is None:
            camera_name = process_pose_data.honeycomb_io.fetch_camera_names([camera_id])[camera_id]
        camera_identifier = camera_name
    fig_suptitle = '{} ({})'.format(
        camera_identifier,
        timestamp.strftime(plot_title_datetime_format)
    )
    save_filename = '{}_{}_{}.{}'.format(
        filename_prefix,
        slugify.slugify(camera_identifier),
        timestamp.strftime(filename_datetime_format),
        filename_extension
    )
    df = df.sort_index()
    pose_ids = df.index.tolist()
    if pose_label_map is None:
        if 'track_label_2d' in df.columns:
            pose_labels = df['track_label_2d'].values.tolist()
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
            pose_model = process_pose_data.honeycomb_io.fetch_pose_model(
                pose_id=pose_ids[0]
            )
            keypoint_connectors = pose_model.get('keypoint_connectors')
    for pose_id, row in df.iterrows():
        process_pose_data.draw_pose_2d(
            row['keypoint_coordinates_2d'],
            draw_keypoint_connectors=draw_keypoint_connectors,
            keypoint_connectors=keypoint_connectors,
            pose_label=pose_label_map[pose_id],
            pose_color=pose_color_map[pose_id],
            keypoint_alpha=keypoint_alpha,
            keypoint_connector_alpha=keypoint_connector_alpha,
            keypoint_connector_linewidth=keypoint_connector_linewidth,
            pose_label_color=pose_label_color,
            pose_label_background_alpha=pose_label_background_alpha,
            pose_label_boxstyle=pose_label_boxstyle
        )
    if show_background_image:
        if background_image is None:
            image_metadata = video_io.fetch_images(
                image_timestamps = [timestamp.to_pydatetime()],
                camera_device_ids=[camera_id]
            )
            image_local_path = image_metadata[0]['image_local_path']
            background_image = cv_utils.fetch_image_from_local_drive(image_local_path)
        cv_utils.draw_background_image(
            image=background_image,
            alpha=background_image_alpha
        )
    cv_utils.format_2d_image_plot(
        image_size=image_size,
        show_axes=show_axes
    )
    fig = plt.gcf()
    fig.suptitle(fig_suptitle)
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    # Show plot
    if show:
        plt.show()
    # Save plot
    if save:
        path = os.path.join(
            save_directory,
            save_filename
        )
        fig.savefig(path)

def draw_poses_3d_timestamp_camera(
    df,
    camera_ids,
    pose_model_id=None,
    camera_names=None,
    camera_calibrations=None,
    keypoint_connectors=None,
    edge_threshold=None,
    keypoint_alpha=0.3,
    keypoint_connector_alpha=0.3,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.5,
    pose_label_boxstyle='circle',
    background_image_alpha=0.4,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S.%f',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='poses_3d',
    filename_datetime_format='%Y%m%d_%H%M%S_%f',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    timestamps = df['timestamp'].unique()
    if len(timestamps) > 1:
        raise ValueError('More than one timestamp in data frame')
    timestamp = timestamps[0]
    if pose_model_id is not None:
        pose_model = process_pose_data.honeycomb_io.fetch_pose_model_by_pose_model_id(
            pose_model_id
        )
        if keypoint_connectors is None:
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if camera_names is None:
        camera_names = process_pose_data.honeycomb_io.fetch_camera_names(
            camera_ids
        )
    if camera_calibrations is None:
        camera_calibrations = process_pose_data.honeycomb_io.fetch_camera_calibrations(
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
        if edge_threshold is not None:
            axis_title = '{} ({} edges) ({})'.format(
                camera_name,
                edge_threshold,
                timestamp.strftime(plot_title_datetime_format)
            )
            save_filename = '{}_{}_{}_{}_edges.{}'.format(
                filename_prefix,
                slugify.slugify(camera_name),
                timestamp.strftime(filename_datetime_format),
                edge_threshold,
                filename_extension
            )
        else:
            axis_title = '{} ({})'.format(
                camera_name,
                timestamp.strftime(plot_title_datetime_format)
            )
            save_filename = '{}_{}_{}.{}'.format(
                filename_prefix,
                slugify.slugify(camera_name),
                timestamp.strftime(filename_datetime_format),
                filename_extension
            )
        image_metadata = video_io.fetch_images(
            image_timestamps = [timestamp.to_pydatetime()],
            camera_device_ids=[camera_id]
        )
        image_local_path = image_metadata[0]['image_local_path']
        background_image = cv_utils.fetch_image_from_local_drive(image_local_path)
        fig, ax = plt.subplots()
        ax.imshow(
            cv.cvtColor(background_image, cv.COLOR_BGR2RGB),
            alpha=background_image_alpha
        )
        for pose_id_3d, row in df.iterrows():
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
            draw_pose_2d(
                keypoint_coordinates=keypoint_coordinates_2d,
                draw_keypoint_connectors=draw_keypoint_connectors,
                keypoint_connectors=keypoint_connectors,
                pose_label=row['match_group_label'],
                pose_color=color_mapping[row['match_group_label']],
                keypoint_alpha=keypoint_alpha,
                keypoint_connector_alpha=keypoint_connector_alpha,
                keypoint_connector_linewidth=keypoint_connector_linewidth,
                pose_label_color=pose_label_color,
                pose_label_background_alpha=pose_label_background_alpha,
                pose_label_boxstyle=pose_label_boxstyle
            )
        ax.axis(
            xmin=0,
            xmax=camera_calibrations[camera_id]['image_width'],
            ymin=camera_calibrations[camera_id]['image_height'],
            ymax=0
        )
        ax.axis('off')
        ax.set_title(axis_title)
        fig.set_size_inches(fig_width_inches, fig_height_inches)
        # Show plot
        if show:
            plt.show()
        # Save plot
        if save:
            path = os.path.join(
                save_directory,
                save_filename
            )
            fig.savefig(path)

def draw_poses_3d_consecutive_timestamps(
    poses_3d_df,
    timestamp,
    camera_ids,
    pose_label_column='match_group_label',
    pose_model_id=None,
    camera_names=None,
    camera_calibrations=None,
    keypoint_connectors=None,
    keypoint_alpha=0.3,
    keypoint_connector_alpha=0.3,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.5,
    pose_label_boxstyle='circle',
    background_image_alpha=0.4,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S.%f',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='draw_poses_3d_consecutive_timestamps',
    filename_datetime_format='%Y%m%d_%H%M%S_%f',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=16
):
    timestamp_previous = timestamp - pd.Timedelta(100, 'ms')
    if pose_model_id is not None:
        pose_model = process_pose_data.honeycomb_io.fetch_pose_model_by_pose_model_id(
            pose_model_id
        )
        if keypoint_connectors is None:
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if camera_names is None:
        camera_names = process_pose_data.honeycomb_io.fetch_camera_names(
            camera_ids
        )
    if camera_calibrations is None:
        camera_calibrations = process_pose_data.honeycomb_io.fetch_camera_calibrations(
            camera_ids,
            start=timestamp.to_pydatetime(),
            end=timestamp.to_pydatetime()
        )
    pose_labels = poses_3d_df.loc[poses_3d_df['timestamp'].isin([timestamp, timestamp_previous]), pose_label_column].fillna('NA').unique()
    color_mapping = process_pose_data.visualize.generate_color_mapping_no_sort(pose_labels)
    for camera_id in camera_ids:
        camera_name = camera_names[camera_id]
        camera_calibration = camera_calibrations[camera_id]
        if keypoint_connectors is not None:
            draw_keypoint_connectors = True
        else:
            draw_keypoint_connectors = False
        fig, axes = plt.subplots(2, 1)
        for axis_index, selected_timestamp in enumerate([timestamp_previous, timestamp]):
            image_metadata = video_io.fetch_images(
                image_timestamps = [selected_timestamp.to_pydatetime()],
                camera_device_ids=[camera_id]
            )
            image_local_path = image_metadata[0]['image_local_path']
            background_image = cv_utils.fetch_image_from_local_drive(image_local_path)
            axes[axis_index].imshow(
                cv.cvtColor(background_image, cv.COLOR_BGR2RGB),
                alpha=background_image_alpha
            )
            for pose_id_3d, row in poses_3d_df[poses_3d_df['timestamp'] == selected_timestamp].iterrows():
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
                plt.sca(axes[axis_index])
                if pd.notnull(row[pose_label_column]):
                    pose_color = color_mapping[row[pose_label_column]]
                else:
                    pose_color = color_mapping['NA']
                draw_pose_2d(
                    keypoint_coordinates=keypoint_coordinates_2d,
                    draw_keypoint_connectors=draw_keypoint_connectors,
                    keypoint_connectors=keypoint_connectors,
                    pose_label=row[pose_label_column],
                    pose_color=pose_color,
                    keypoint_alpha=keypoint_alpha,
                    keypoint_connector_alpha=keypoint_connector_alpha,
                    keypoint_connector_linewidth=keypoint_connector_linewidth,
                    pose_label_color=pose_label_color,
                    pose_label_background_alpha=pose_label_background_alpha,
                    pose_label_boxstyle=pose_label_boxstyle
                )
            axes[axis_index].axis(
                xmin=0,
                xmax=camera_calibrations[camera_id]['image_width'],
                ymin=camera_calibrations[camera_id]['image_height'],
                ymax=0
            )
            axes[axis_index].axis('off')
            axes[axis_index].set_title(selected_timestamp.strftime(plot_title_datetime_format))
        fig.set_size_inches(fig_width_inches, fig_height_inches)
        fig.suptitle(camera_names[camera_id])
        # Show plot
        if show:
            plt.show()
        # Save plot
        if save:
            save_filename = '{}_{}_{}.{}'.format(
                filename_prefix,
                slugify.slugify(camera_name),
                timestamp.strftime(filename_datetime_format),
                filename_extension
            )
            path = os.path.join(
                save_directory,
                save_filename
            )
            fig.savefig(path)

def draw_pose_2d(
    keypoint_coordinates,
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    pose_label=None,
    pose_color=None,
    keypoint_alpha=0.3,
    keypoint_connector_alpha=0.3,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.5,
    pose_label_boxstyle='circle'
):
    keypoint_coordinates = np.asarray(keypoint_coordinates).reshape((-1, 2))
    valid_keypoints = np.all(np.isfinite(keypoint_coordinates), axis=1)
    plottable_points = keypoint_coordinates[valid_keypoints]
    points_image_u = plottable_points[:, 0]
    points_image_v = plottable_points[:, 1]
    plot, = plt.plot(
        points_image_u,
        points_image_v,
        '.',
        color=pose_color,
        alpha=keypoint_alpha)
    plot_color=color=plot.get_color()
    if draw_keypoint_connectors and (keypoint_connectors is not None):
        for keypoint_connector in keypoint_connectors:
            keypoint_from_index = keypoint_connector[0]
            keypoint_to_index = keypoint_connector[1]
            if valid_keypoints[keypoint_from_index] and valid_keypoints[keypoint_to_index]:
                plt.plot(
                    [keypoint_coordinates[keypoint_from_index,0],keypoint_coordinates[keypoint_to_index, 0]],
                    [keypoint_coordinates[keypoint_from_index,1],keypoint_coordinates[keypoint_to_index, 1]],
                    linewidth=keypoint_connector_linewidth,
                    color=plot_color,
                    alpha=keypoint_connector_alpha
                )
    if pose_label is not None:
        pose_label_anchor = np.nanmean(keypoint_coordinates, axis=0)
        plt.text(
            pose_label_anchor[0],
            pose_label_anchor[1],
            pose_label,
            color=pose_label_color,
            bbox={
                'alpha': pose_label_background_alpha,
                'facecolor': plot_color,
                'edgecolor': 'none',
                'boxstyle': pose_label_boxstyle
            }
        )

def visualize_pose_pair(
    pose_pair,
    camera_calibrations=None,
    camera_names=None,
    floor_marker_color='blue',
    floor_marker_linewidth=2,
    floor_marker_linestyle='-',
    floor_marker_alpha=1.0,
    vertical_line_color='blue',
    vertical_line_linewidth=2,
    vertical_line_linestyle='--',
    vertical_line_alpha=1.0,
    centroid_markersize=16,
    centroid_color='red',
    centroid_alpha=1.0,
    pose_label_background_color='red',
    pose_label_background_alpha=1.0,
    pose_label_color='white',
    pose_label_boxstyle='circle',
    background_image_alpha=0.4,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S.%f',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='pose_pair',
    filename_datetime_format='%Y%m%d_%H%M%S_%f',
    filename_extension='png',
    fig_width_inches=8,
    fig_height_inches=10.5
):
    timestamp = pose_pair.get('timestamp')
    camera_id_a = pose_pair.get('camera_id_a')
    camera_id_b = pose_pair.get('camera_id_b')
    if camera_calibrations is None:
        camera_calibrations = process_pose_data.fetch_camera_calibrations(
            camera_ids=[camera_id_a, camera_id_b],
            start=timestamp.to_pydatetime(),
            end=timestamp.to_pydatetime()
        )
    if camera_names is None:
        camera_names = process_pose_data.fetch_camera_names(
            camera_ids=[camera_id_a, camera_id_b]
        )
    fig_suptitle = timestamp.strftime(plot_title_datetime_format)
    save_filename = '{}_{}_{}.{}'.format(
        filename_prefix,
        slugify.slugify(pose_pair['pose_id_a']),
        slugify.slugify(pose_pair['pose_id_b']),
        filename_extension
    )
    centroid_3d = np.nanmean(pose_pair['keypoint_coordinates_3d'], axis=0)
    floor_marker_x_3d = np.array([[x, centroid_3d[1], 0] for x in np.linspace(centroid_3d[0] - 1.0, centroid_3d[0] + 1.0, 100)])
    floor_marker_y_3d = np.array([[centroid_3d[0], y, 0] for y in np.linspace(centroid_3d[1] - 1.0, centroid_3d[1] + 1.0, 100)])
    vertical_line_3d = np.array([[centroid_3d[0], centroid_3d[1], z] for z in np.linspace(0, centroid_3d[2], 100)])
    fig, axes = plt.subplots(2, 1)
    for axis_index, suffix in enumerate(['a', 'b']):
        axis_title = camera_names[pose_pair['camera_id_' + suffix]]
        centroid = cv_utils.project_points(
            object_points=centroid_3d,
            rotation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['rotation_vector'],
            translation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['translation_vector'],
            camera_matrix=camera_calibrations[pose_pair['camera_id_' + suffix]]['camera_matrix'],
            distortion_coefficients=camera_calibrations[pose_pair['camera_id_' + suffix]]['distortion_coefficients'],
            remove_behind_camera=True
        )
        floor_marker_x = cv_utils.project_points(
            object_points=floor_marker_x_3d,
            rotation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['rotation_vector'],
            translation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['translation_vector'],
            camera_matrix=camera_calibrations[pose_pair['camera_id_' + suffix]]['camera_matrix'],
            distortion_coefficients=camera_calibrations[pose_pair['camera_id_' + suffix]]['distortion_coefficients'],
            remove_behind_camera=True
        )
        floor_marker_y = cv_utils.project_points(
            object_points=floor_marker_y_3d,
            rotation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['rotation_vector'],
            translation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['translation_vector'],
            camera_matrix=camera_calibrations[pose_pair['camera_id_' + suffix]]['camera_matrix'],
            distortion_coefficients=camera_calibrations[pose_pair['camera_id_' + suffix]]['distortion_coefficients'],
            remove_behind_camera=True
        )
        vertical_line = cv_utils.project_points(
            object_points=vertical_line_3d,
            rotation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['rotation_vector'],
            translation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['translation_vector'],
            camera_matrix=camera_calibrations[pose_pair['camera_id_' + suffix]]['camera_matrix'],
            distortion_coefficients=camera_calibrations[pose_pair['camera_id_' + suffix]]['distortion_coefficients'],
            remove_behind_camera=True
        )
        image_metadata = video_io.fetch_images(
            image_timestamps = [timestamp.to_pydatetime()],
            camera_device_ids=[pose_pair['camera_id_' + suffix]]
        )
        image_local_path = image_metadata[0]['image_local_path']
        background_image = cv_utils.fetch_image_from_local_drive(image_local_path)
        axes[axis_index].imshow(
            cv.cvtColor(background_image, cv.COLOR_BGR2RGB),
            alpha=background_image_alpha
        )
        if pose_pair.get('track_label_2d_' + suffix) is not None:
            axes[axis_index].text(
                centroid[0],
                centroid[1],
                pose_pair.get('track_label_2d_' + suffix),
                color=pose_label_color,
                bbox={
                    'alpha': pose_label_background_alpha,
                    'facecolor': pose_label_background_color,
                    'edgecolor': 'none',
                    'boxstyle': pose_label_boxstyle
                }
            )
        else:
            axes[axis_index].plot(
                centroid[0],
                centroid[1],
                '.',
                color=centroid_color,
                markersize=centroid_markersize,
                alpha=centroid_alpha
            )
        axes[axis_index].plot(
            floor_marker_x[:, 0],
            floor_marker_x[:, 1],
            color=floor_marker_color,
            linewidth=floor_marker_linewidth,
            linestyle=floor_marker_linestyle,
            alpha=floor_marker_alpha,
        )
        axes[axis_index].plot(
            floor_marker_y[:, 0],
            floor_marker_y[:, 1],
            color=floor_marker_color,
            linewidth=floor_marker_linewidth,
            linestyle=floor_marker_linestyle,
            alpha=floor_marker_alpha,
        )
        axes[axis_index].plot(
            vertical_line[:, 0],
            vertical_line[:, 1],
            color=vertical_line_color,
            linewidth=vertical_line_linewidth,
            linestyle=vertical_line_linestyle,
            alpha=vertical_line_alpha,
        )
        axes[axis_index].axis(
            xmin=0,
            xmax=camera_calibrations[pose_pair['camera_id_' + suffix]]['image_width'],
            ymin=camera_calibrations[pose_pair['camera_id_' + suffix]]['image_height'],
            ymax=0
        )
        axes[axis_index].axis('off')
        axes[axis_index].set_title(axis_title)
    fig.suptitle(fig_suptitle)
    plt.subplots_adjust(top=0.95, hspace=0.1)
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    plt.show()
    # Show plot
    if show:
        plt.show()
    # Save plot
    if save:
        path = os.path.join(
            save_directory,
            save_filename
        )
        fig.savefig(path)

def extract_single_camera_data(
    df,
):
    single_camera_columns = list()
    for column in df.columns:
        if column[-2:]=='_a' or column[-2:]=='_b':
            single_camera_column = column[:-2]
            if single_camera_column not in single_camera_columns:
                single_camera_columns.append(single_camera_column)
    df = df.reset_index()
    dfs = dict()
    for camera_letter in ['a', 'b']:
        extraction_columns = ['pose_id_' + camera_letter, 'timestamp']
        extraction_columns.extend([single_camera_column + '_' + camera_letter for single_camera_column in single_camera_columns])
        target_columns = ['pose_id', 'timestamp']
        target_columns.extend(single_camera_columns)
        column_map = dict(zip(extraction_columns, target_columns))
        df_single_camera = df.reindex(columns=extraction_columns)
        df_single_camera.rename(columns=column_map, inplace=True)
        df_single_camera.drop_duplicates(subset='pose_id', inplace=True)
        df_single_camera.set_index('pose_id', inplace=True)
        dfs[camera_letter] = df_single_camera
    return dfs

def concat_videos(
    input_videos_path_list,
    output_video_path,
    delete_input_videos=True
):
    temp_file_list_path = './temp_file_list.txt'
    fp = open(temp_file_list_path, 'w')
    for input_video_path in input_videos_path_list:
        if not os.path.isfile(input_video_path):
            fp.close()
            os.remove(temp_file_list_path)
            raise ValueError('Input video file {} does not exist'.format(input_video_path))
        fp.write('file {}\n'.format(input_video_path))
    fp.close()
    stream  = ffmpeg.input(temp_file_list_path, format='concat', safe=0)
    stream = ffmpeg.output(stream, output_video_path, c='copy')
    ffmpeg.run(stream)
    if delete_input_videos:
        for input_video_path in input_videos_path_list:
            os.remove(input_video_path)
    os.remove(temp_file_list_path)
