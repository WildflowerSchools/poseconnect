import process_pose_data.local_io
import process_pose_data.visualize
import honeycomb_io
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
import functools
import datetime
import string
import logging
import multiprocessing
import os

logger = logging.getLogger(__name__)

def overlay_poses(
    poses_df,
    start=None,
    end=None,
    camera_assignment_ids=None,
    environment_id=None,
    environment_name=None,
    camera_device_types=None,
    camera_device_ids=None,
    camera_part_numbers=None,
    camera_names=None,
    camera_serial_numbers=None,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    local_video_directory='./videos',
    video_filename_extension='mp4',
    pose_model_id=None,
    camera_calibrations=None,
    pose_label_column=None,
    keypoint_connectors=None,
    pose_color='green',
    keypoint_radius=3,
    keypoint_alpha=0.6,
    keypoint_connector_alpha=0.6,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.6,
    pose_label_font_scale=1.5,
    pose_label_line_width=1,
    output_directory='./video_overlays',
    output_filename_prefix='poses',
    output_filename_datetime_format='%Y%m%d_%H%M%S_%f',
    output_filename_extension='avi',
    output_fourcc_string='XVID',
    concatenate_videos=True,
    delete_individual_clips=True,
    parallel=False,
    num_parallel_processes=None,
    task_progress_bar=False,
    segment_progress_bar=False,
    notebook=False
):
    poses_df_columns = poses_df.columns.tolist()
    if 'keypoint_coordinates_3d' in poses_df_columns and 'camera_id' not in poses_df_columns:
        logger.info('Poses appear to be 3D poses. Projecting into each camera view')
        poses_3d = True
        poses_2d = False
    elif 'keypoint_coordinates_3d' not in poses_df_columns and 'camera_id' in poses_df_columns:
        logger.info('Poses appear to be 2D poses. Subsetting poses foreach camera view')
        poses_3d = False
        poses_2d = True
    elif len(poses_df) == 0:
        logger.warn('Pose dataframe is empty')
        return
    else:
        raise ValueError('Cannot parse pose dataframe')
    video_metadata_with_local_paths = video_io.fetch_videos(
        start=start,
        end=end,
        video_timestamps=None,
        camera_assignment_ids=camera_assignment_ids,
        environment_id=environment_id,
        environment_name=environment_name,
        camera_device_types=camera_device_types,
        camera_device_ids=camera_device_ids,
        camera_part_numbers=camera_part_numbers,
        camera_names=camera_names,
        camera_serial_numbers=camera_serial_numbers,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret,
        local_video_directory=local_video_directory,
        video_filename_extension=video_filename_extension
    )
    video_metadata_dict = dict()
    for datum in video_metadata_with_local_paths:
        camera_id = datum.get('device_id')
        video_timestamp = datum.get('video_timestamp')
        if camera_id not in video_metadata_dict.keys():
            video_metadata_dict[camera_id] = dict()
        video_metadata_dict[camera_id][video_timestamp] = datum
    camera_ids = list(video_metadata_dict.keys())
    camera_name_dict = honeycomb_io.fetch_camera_names(
        camera_ids
    )
    if poses_3d:
        if camera_calibrations is None:
            camera_calibrations = honeycomb_io.fetch_camera_calibrations(
                camera_ids,
                start=start,
                end=end
            )
    if pose_model_id is not None:
        pose_model = honeycomb_io.fetch_pose_model_by_pose_model_id(
            pose_model_id
        )
        if keypoint_connectors is None:
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if keypoint_connectors is not None:
        draw_keypoint_connectors = True
    else:
        draw_keypoint_connectors = False
    overlay_poses_camera_time_segment_partial = functools.partial(
        overlay_poses_camera_time_segment,
        video_metadata_dict=video_metadata_dict,
        camera_name_dict=camera_name_dict,
        pose_label_column=pose_label_column,
        draw_keypoint_connectors=draw_keypoint_connectors,
        keypoint_connectors=keypoint_connectors,
        pose_color=pose_color,
        keypoint_radius=keypoint_radius,
        keypoint_alpha=keypoint_alpha,
        keypoint_connector_alpha=keypoint_connector_alpha,
        keypoint_connector_linewidth=keypoint_connector_linewidth,
        pose_label_color=pose_label_color,
        pose_label_background_alpha=pose_label_background_alpha,
        pose_label_font_scale=pose_label_font_scale,
        pose_label_line_width=pose_label_line_width,
        output_directory=output_directory,
        output_filename_prefix=output_filename_prefix,
        output_filename_datetime_format=output_filename_datetime_format,
        output_filename_extension=output_filename_extension,
        output_fourcc_string=output_fourcc_string,
        progress_bar=segment_progress_bar,
        notebook=notebook
    )
    input_parameters_list = list()
    output_parameters_list = list()
    for camera_id in camera_ids:
        if poses_3d:
            camera_calibration = camera_calibrations[camera_id]
            project_points_partial = functools.partial(
                cv_utils.project_points,
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
        logger.info('Overlaying poses for {}'.format(camera_name_dict[camera_id]))
        video_timestamps = sorted(video_metadata_dict[camera_id].keys())
        for video_timestamp in video_timestamps:
            # Add an extra second to capture extra frames in video
            poses_time_segment_df = poses_df.loc[
                (poses_df['timestamp'] >= video_timestamp )&
                (poses_df['timestamp'] < video_timestamp + datetime.timedelta(seconds=11))
            ].copy()
            if poses_3d:
                poses_time_segment_df['keypoint_coordinates_2d'] = poses_time_segment_df['keypoint_coordinates_3d'].apply(project_points_partial)
            if poses_2d:
                poses_time_segment_df = poses_time_segment_df.loc[poses_time_segment_df['camera_id'] == camera_id].copy()
            input_parameters_list.append({
                'poses_df': poses_time_segment_df,
                'camera_id': camera_id,
                'video_timestamp': video_timestamp
            })
    if parallel:
        logger.info('Attempting to launch parallel processes')
        if num_parallel_processes is None:
            num_cpus=multiprocessing.cpu_count()
            num_processes = num_cpus - 1
            logger.info('Number of parallel processes not specified. {} CPUs detected. Launching {} processes'.format(
                num_cpus,
                num_processes
            ))
        with multiprocessing.Pool(num_processes) as p:
            if task_progress_bar:
                if notebook:
                    output_parameters_list = list(tqdm.notebook.tqdm(
                        p.imap(
                            overlay_poses_camera_time_segment_partial,
                            input_parameters_list
                        ),
                        total=len(input_parameters_list)
                    ))
                else:
                    output_parameters_list = list(tqdm.tqdm(
                        p.imap(
                            overlay_poses_camera_time_segment_partial,
                            input_parameters_list
                        ),
                        total=len(input_parameters_list)
                    ))
            else:
                output_parameters_list = list(
                    p.imap(
                        overlay_poses_camera_time_segment_partial,
                        input_parameters_list
                    )
                )
    else:
        if task_progress_bar:
            if notebook:
                output_parameters_list = list(map(
                    overlay_poses_camera_time_segment_partial,
                    tqdm.notebook.tqdm(input_parameters_list)
                ))
            else:
                output_parameters_list = list(map(
                    overlay_poses_camera_time_segment_partial,
                    tqdm.tqdm(input_parameters_list)
                ))
        else:
            output_parameters_list = list(map(
                overlay_poses_camera_time_segment_partial,
                input_parameters_list
            ))
    if concatenate_videos:
        output_path_dict = dict()
        for output_parameters in output_parameters_list:
            camera_id = output_parameters['camera_id']
            output_path = output_parameters['output_path']
            if camera_id not in output_path_dict.keys():
                output_path_dict[camera_id] = list()
            output_path_dict[camera_id].append(output_path)
        for camera_id, output_paths in output_path_dict.items():
            concat_output_path = os.path.join(
                output_directory,
                '{}_{}_{}_{}.{}'.format(
                    output_filename_prefix,
                    video_timestamps[0].strftime(output_filename_datetime_format),
                    video_timestamps[-1].strftime(output_filename_datetime_format),
                    slugify.slugify(camera_name_dict[camera_id]),
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

def overlay_poses_camera_time_segment(
    input_parameters,
    video_metadata_dict,
    camera_name_dict,
    pose_label_column=None,
    draw_keypoint_connectors=False,
    keypoint_connectors=None,
    pose_color='green',
    keypoint_radius=3,
    keypoint_alpha=0.6,
    keypoint_connector_alpha=0.6,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.6,
    pose_label_font_scale=1.5,
    pose_label_line_width=1,
    output_directory='.',
    output_filename_prefix='poses_overlay',
    output_filename_datetime_format='%Y%m%d_%H%M%S_%f',
    output_filename_extension='mp4',
    output_fourcc_string=None,
    progress_bar=False,
    notebook=False
):
    poses_df = input_parameters['poses_df']
    camera_id = input_parameters['camera_id']
    video_timestamp = input_parameters['video_timestamp']
    camera_name = camera_name_dict[camera_id]
    video_local_path = video_metadata_dict[camera_id][video_timestamp]['video_local_path']
    logger.info('Overlaying poses for video starting at {}'.format(video_timestamp.isoformat()))
    logger.info('Video input path: {}'.format(video_local_path))
    output_path = os.path.join(
        output_directory,
        '{}_{}_{}.{}'.format(
            output_filename_prefix,
            video_timestamp.strftime(output_filename_datetime_format),
            slugify.slugify(camera_name),
            output_filename_extension
        )
    )
    logger.info('Video output path: {}'.format(output_path))
    video_input = cv_utils.VideoInput(
        input_path=video_local_path,
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
    if pd.to_datetime(video_start_time, utc=True) > poses_df['timestamp'].max():
        raise ValueError('Video starts at {} but 3D pose data ends at {}'.format(
            video_start_time.isoformat(),
            poses_df['timestamp'].max().isoformat()
        ))
    video_end_time = video_start_time + datetime.timedelta(milliseconds=(video_frame_count - 1)*100)
    if pd.to_datetime(video_end_time, utc=True) < poses_df['timestamp'].min():
        raise ValueError('Video ends at {} but 3D pose data starts at {}'.format(
            video_end_time.isoformat(),
            poses_df['timestamp'].min().isoformat()
        ))
    video_output_parameters = video_input.video_parameters
    if output_fourcc_string is not None:
        video_output_parameters.fourcc_int = cv_utils.fourcc_string_to_int(output_fourcc_string)
    os.makedirs(output_directory, exist_ok=True)
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
        for pose_id, row in poses_df.loc[poses_df['timestamp'] == timestamp_pandas].iterrows():
            keypoint_coordinates_2d = row['keypoint_coordinates_2d']
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
    output_parameters = {
        'camera_id': camera_id,
        'video_timestamp': video_timestamp,
        'output_path': output_path
    }
    return output_parameters

def visualize_3d_pose_reconstruction(
    pose_3d_id,
    pose_model_id,
    floor_height=0.0,
    poses_3d_df=None,
    pose_reconstruction_3d_inference_id=None,
    pose_3d_timestamp=None,
    poses_2d_df=None,
    pose_extraction_2d_inference_id=None,
    base_dir=None,
    environment_id=None,
    pose_processing_subdirectory='pose_processing',
    camera_device_id_lookup=None,
    camera_calibrations=None,
    camera_names=None,
    local_image_directory='./images',
    image_filename_extension='png',
    local_video_directory='./videos',
    video_filename_extension='mp4',
    pose_3d_color='green',
    pose_3d_footprint_color='green',
    pose_3d_footprint_alpha=0.5,
    constituent_pose_2d_color='blue',
    non_constituent_pose_2d_color='red',
    keypoint_connectors=None,
    keypoint_radius=3,
    keypoint_alpha=0.6,
    keypoint_connector_alpha=0.6,
    keypoint_connector_linewidth=3,
    timestamp_text_color='white',
    timestamp_background_color='black',
    timestamp_background_alpha=0.3,
    timestamp_font_scale=1.5,
    timestamp_line_width=1,
    output_directory='./image_overlays',
    output_filename_stem='pose_3d_reconstruction',
    output_filename_extension='png',
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    pass
    # Fetch 3D poses (if necessary)
    if poses_3d_df is None:
        if (
            base_dir is None or
            environment_id is None or
            pose_reconstruction_3d_inference_id is None or
            pose_3d_timestamp is None
        ):
            raise ValueError('If poses_3d_df not supplied, must specify base_dir, and environment_id, pose_reconstruction_3d_inference_id, and pose_3d_timestamp')
        poses_3d_df = process_pose_data.local_io.fetch_data_local_by_time_segment(
            start=pose_3d_timestamp,
            end=pose_3d_timestamp,
            base_dir=base_dir,
            pipeline_stage='pose_reconstruction_3d',
            environment_id=environment_id,
            filename_stem='poses_3d',
            inference_ids=pose_reconstruction_3d_inference_id,
            data_ids=[pose_3d_id],
            sort_field=None,
            object_type='dataframe',
            pose_processing_subdirectory='pose_processing'
        )
    # Extract 3D pose
    if pose_3d_id not in poses_3d_df.index:
        raise ValueError('3D pose {} not found in 3D poses'.format(
            pose_3d_id
        ))
    pose_3d = poses_3d_df.loc[pose_3d_id]
    pose_3d_timestamp = pose_3d['timestamp']
    pose_3d_keypoint_coordinates_3d = pose_3d['keypoint_coordinates_3d']
    pose_3d_pose_2d_ids = pose_3d['pose_2d_ids']
    # Calculate 3D pose footprint
    x_min = np.nanmin(pose_3d_keypoint_coordinates_3d[:, 0])
    x_max = np.nanmax(pose_3d_keypoint_coordinates_3d[:, 0])
    y_min = np.nanmin(pose_3d_keypoint_coordinates_3d[:, 1])
    y_max = np.nanmax(pose_3d_keypoint_coordinates_3d[:, 1])
    pose_3d_footprint_vertices_3d = np.array([
        [x_min, y_min, floor_height],
        [x_min, y_max, floor_height],
        [x_max, y_max, floor_height],
        [x_max, y_min, floor_height]
    ])
    # Fetch 2D poses (if necessary)
    if poses_2d_df is None:
        if (
            base_dir is None or
            environment_id is None or
            pose_extraction_2d_inference_id is None or
            pose_3d_timestamp is None
        ):
            raise ValueError('If poses_2d_df not supplied, must specify base_dir, environment_id, pose_extraction_2d_inference_id, and pose_3d_timestamp')
        poses_2d_df = process_pose_data.local_io.fetch_data_local_by_time_segment(
            start=pose_3d_timestamp,
            end=pose_3d_timestamp,
            base_dir=base_dir,
            pipeline_stage='pose_extraction_2d',
            environment_id=environment_id,
            filename_stem='poses_2d',
            inference_ids=pose_extraction_2d_inference_id,
            data_ids=None,
            sort_field=None,
            object_type='dataframe',
            pose_processing_subdirectory='pose_processing'
        )
    # Extract 2D poses from the same timestamp as the 3D pose
    poses_2d_df = poses_2d_df.loc[poses_2d_df['timestamp'] == pose_3d_timestamp].copy()
    if len(poses_2d_df) == 0:
        raise ValueError('Timestamp associated with 3D pose {} not found in 2D poses'.format(
            pose_3d_timestamp.isoformat()
        ))
    # Convert assignment IDs to device IDs for cameras
    poses_2d_df = process_pose_data.local_io.convert_assignment_ids_to_camera_device_ids(
        poses_2d_df=poses_2d_df,
        camera_device_id_lookup=camera_device_id_lookup,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    camera_ids = poses_2d_df['camera_id'].unique().tolist()
    # Fetch camera names (if necessary)
    if camera_names is None:
        camera_names = honeycomb_io.fetch_camera_names(
            camera_ids=camera_ids,
            chunk_size=chunk_size,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    # Fetch camera calibrations (if necessary)
    if camera_calibrations is None:
        camera_calibrations = honeycomb_io.fetch_camera_calibrations(
            camera_ids=camera_ids,
            start=pose_3d_timestamp,
            end=pose_3d_timestamp,
            chunk_size=chunk_size,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    # Download images
    image_metadata_with_local_paths = video_io.fetch_images(
        image_timestamps=[pose_3d_timestamp],
        camera_device_ids=camera_ids,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret,
        local_image_directory=local_image_directory,
        image_filename_extension=image_filename_extension,
        local_video_directory=local_video_directory,
        video_filename_extension=video_filename_extension
    )
    image_metadata_dict = {
        image_metadatum['device_id']: image_metadatum
        for image_metadatum in image_metadata_with_local_paths
    }
    # Fetch information for keypoint connectors (if necessary)
    if pose_model_id is not None:
        pose_model = honeycomb_io.fetch_pose_model_by_pose_model_id(
            pose_model_id
        )
        if keypoint_connectors is None:
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if keypoint_connectors is not None:
        draw_keypoint_connectors = True
    else:
        draw_keypoint_connectors = False
    # Create overlays
    for camera_id in camera_ids:
        # Project 3D pose into this camera
        pose_3d_keypoint_coordinates_2d = cv_utils.project_points(
            object_points=pose_3d_keypoint_coordinates_3d,
            rotation_vector = camera_calibrations[camera_id]['rotation_vector'],
            translation_vector = camera_calibrations[camera_id]['translation_vector'],
            camera_matrix = camera_calibrations[camera_id]['camera_matrix'],
            distortion_coefficients = camera_calibrations[camera_id]['distortion_coefficients'],
            remove_behind_camera=True,
            remove_outside_frame=True,
            image_corners=[
                [0,0],
                [camera_calibrations[camera_id]['image_width'], camera_calibrations[camera_id]['image_height']]
            ]
        )
        # Project 3D pose footprint into this camera
        pose_3d_footprint_vertices_2d = cv_utils.project_points(
            object_points=pose_3d_footprint_vertices_3d,
            rotation_vector=camera_calibrations[camera_id]['rotation_vector'],
            translation_vector=camera_calibrations[camera_id]['translation_vector'],
            camera_matrix=camera_calibrations[camera_id]['camera_matrix'],
            distortion_coefficients=camera_calibrations[camera_id]['distortion_coefficients'],
            remove_behind_camera=True,
            remove_outside_frame=True,
            image_corners=[
                [0,0],
                [camera_calibrations[camera_id]['image_width'], camera_calibrations[camera_id]['image_height']]
            ]
        )
        # Load image into memory
        image = cv.imread(image_metadata_dict.get(camera_id).get('image_local_path'))
        # Draw projected 3D pose in one color
        image = draw_pose_2d_opencv(
            image=image,
            keypoint_coordinates=pose_3d_keypoint_coordinates_2d,
            draw_keypoint_connectors=draw_keypoint_connectors,
            keypoint_connectors=keypoint_connectors,
            pose_label=None,
            pose_color=pose_3d_color,
            keypoint_radius=keypoint_radius,
            keypoint_alpha=keypoint_alpha,
            keypoint_connector_alpha=keypoint_connector_alpha,
            keypoint_connector_linewidth=keypoint_connector_linewidth
        )
        # Draw floor box underneath 3D pose in same color color
        if np.all(np.isfinite(pose_3d_footprint_vertices_2d)):
            image = draw_polygon_2d_opencv(
                image=image,
                vertices=pose_3d_footprint_vertices_2d,
                fill_color=pose_3d_footprint_color,
                fill_alpha=pose_3d_footprint_alpha
            )
        # Draw 2D poses
        poses_2d_camera_df = poses_2d_df.loc[poses_2d_df['camera_id'] == camera_id]
        for pose_2d_id, pose_2d in poses_2d_camera_df.iterrows():
            if pose_2d_id in pose_3d_pose_2d_ids:
                pose_2d_color = constituent_pose_2d_color
            else:
                pose_2d_color = non_constituent_pose_2d_color
            image = draw_pose_2d_opencv(
                image=image,
                keypoint_coordinates=pose_2d['keypoint_coordinates_2d'],
                draw_keypoint_connectors=draw_keypoint_connectors,
                keypoint_connectors=keypoint_connectors,
                pose_label=None,
                pose_color=pose_2d_color,
                keypoint_radius=keypoint_radius,
                keypoint_alpha=keypoint_alpha,
                keypoint_connector_alpha=keypoint_connector_alpha,
                keypoint_connector_linewidth=keypoint_connector_linewidth
            )
        # Write timestamp in the corner
        image = draw_timestamp_opencv(
            image=image,
            timestamp=pose_3d_timestamp,
            text_color=timestamp_text_color,
            background_color=timestamp_background_color,
            background_alpha=timestamp_background_alpha,
            font_scale=timestamp_font_scale,
            line_width=timestamp_line_width
        )
        # Build image save path
        output_path = os.path.join(
            output_directory,
            '{}_{}_{}.{}'.format(
                output_filename_stem,
                pose_3d_id,
                slugify.slugify(camera_names[camera_id]),
                output_filename_extension
            )
        )
        # Create directory if necessary
        os.makedirs(output_directory, exist_ok=True)
        # Save image
        cv.imwrite(output_path, image)


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
    pose_label_font_scale=1.5,
    pose_label_line_width=1,
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
            pose_2d_ids = dfs_single_camera[camera_letter].index.values.tolist()
            pose_label_maps[camera_letter] = {pose_2d_id: '' for pose_2d_id in pose_2d_ids}
            if not generate_match_aliases:
                if 'track_label_2d' in dfs_single_camera[camera_letter].columns:
                    pose_labels = dfs_single_camera[camera_letter]['track_label_2d'].values.tolist()
                elif len(pose_2d_ids) <= 26:
                    pose_labels = string.ascii_uppercase[:len(pose_2d_ids)]
                else:
                    pose_labels = range(len(pose_2d_ids))
                pose_label_maps[camera_letter] = dict(zip(pose_2d_ids, pose_labels))
            pose_color_maps[camera_letter] = {pose_2d_id: 'grey' for pose_2d_id in pose_2d_ids}
        match_aliases = iter(list(string.ascii_uppercase[:num_matches]))
        match_colors = iter(sns.color_palette('husl', n_colors=num_matches))
        for (pose_2d_id_a, pose_2d_id_b), row in df.iterrows():
            if row['match']:
                old_label_a = pose_label_maps['a'][pose_2d_id_a]
                old_label_b = pose_label_maps['b'][pose_2d_id_b]
                pose_label_maps['a'][pose_2d_id_a] = '{} ({})'.format(
                    old_label_a,
                    old_label_b
                )
                pose_label_maps['b'][pose_2d_id_b] = '{} ({})'.format(
                    old_label_b,
                    old_label_a
                )
                if generate_match_aliases:
                    match_alias = next(match_aliases)
                    pose_label_maps['a'][pose_2d_id_a] = match_alias
                    pose_label_maps['b'][pose_2d_id_b] = match_alias
                pose_color = next(match_colors)
                pose_color_maps['a'][pose_2d_id_a] = pose_color
                pose_color_maps['b'][pose_2d_id_b] = pose_color
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
    pose_label_font_scale=1.5,
    pose_label_line_width=1,
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
    pose_2d_ids = df.index.tolist()
    if pose_label_map is None:
        if 'track_label_2d' in df.columns:
            pose_labels = df['track_label_2d'].values.tolist()
        elif len(pose_2d_ids) <= 26:
            pose_labels = string.ascii_uppercase[:len(pose_2d_ids)]
        else:
            pose_labels = range(len(pose_2d_ids))
        pose_label_map = dict(zip(pose_2d_ids, pose_labels))
    if pose_color_map is None:
        pose_colors = sns.color_palette('husl', n_colors=len(pose_2d_ids))
        pose_color_map = dict(zip(pose_2d_ids, pose_colors))
    if draw_keypoint_connectors:
        if keypoint_connectors is None:
            pose_model = honeycomb_io.fetch_pose_model(
                pose_2d_id=pose_2d_ids[0]
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
    for pose_2d_id, row in df.iterrows():
        new_image = draw_pose_2d_opencv(
            image=new_image,
            keypoint_coordinates=row['keypoint_coordinates_2d'],
            draw_keypoint_connectors=draw_keypoint_connectors,
            keypoint_connectors=keypoint_connectors,
            pose_label=pose_label_map[pose_2d_id],
            pose_color=pose_color_map[pose_2d_id],
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
    pose_label_font_scale=1.5,
    pose_label_line_width=1
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
    if pd.notna(pose_label):
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

def draw_polygon_2d_opencv(
    image,
    vertices,
    fill_color='white',
    fill_alpha=0.5
):
    fill_color = matplotlib.colors.to_hex(fill_color, keep_alpha=False)
    image = cv_utils.draw_polygon(
        original_image=image,
        vertices=vertices,
        color=fill_color,
        alpha=fill_alpha
    )
    return image

def draw_timestamp_opencv(
    image,
    timestamp,
    text_color='white',
    background_color='black',
    background_alpha=0.3,
    font_scale=1.5,
    line_width=1,
    padding = 5
):
    text_color = matplotlib.colors.to_hex(text_color, keep_alpha=False)
    background_color = matplotlib.colors.to_hex(background_color, keep_alpha=False)
    image_height, image_width, image_depth = image.shape
    upper_right_coordinates = [image_width - padding, padding]
    timestamp_text = timestamp.isoformat()
    text_box_size, baseline = cv.getTextSize(
        text=str(timestamp_text),
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=font_scale,
        thickness=line_width
    )
    image=cv_utils.draw_rectangle(
        original_image=image,
        coordinates=[
            [
                upper_right_coordinates[0] - text_box_size[0],
                upper_right_coordinates[1]
            ],
            [
                upper_right_coordinates[0],
                upper_right_coordinates[1] + text_box_size[1]
            ]
        ],
        line_width=0,
        color=background_color,
        fill=True,
        alpha=background_alpha
    )
    image = cv_utils.draw_text(
        original_image=image,
        coordinates=upper_right_coordinates,
        text=timestamp.isoformat(),
        horizontal_alignment='right',
        vertical_alignment='top',
        font_face=cv.FONT_HERSHEY_PLAIN,
        font_scale=font_scale,
        line_width=line_width,
        color=text_color
    )
    return image

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
        pose_model = honeycomb_io.fetch_pose_model_by_pose_model_id(
            pose_model_id
        )
        if keypoint_connectors is None:
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if camera_names is None:
        camera_names = honeycomb_io.fetch_camera_names(
            camera_ids
        )
    if camera_calibrations is None:
        camera_calibrations = honeycomb_io.fetch_camera_calibrations(
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
            pose_2d_ids = dfs_single_camera[camera_letter].index.values.tolist()
            pose_label_maps[camera_letter] = {pose_2d_id: '' for pose_2d_id in pose_2d_ids}
            if not generate_match_aliases:
                if 'track_label_2d' in dfs_single_camera[camera_letter].columns:
                    pose_labels = dfs_single_camera[camera_letter]['track_label_2d'].values.tolist()
                elif len(pose_2d_ids) <= 26:
                    pose_labels = string.ascii_uppercase[:len(pose_2d_ids)]
                else:
                    pose_labels = range(len(pose_2d_ids))
                pose_label_maps[camera_letter] = dict(zip(pose_2d_ids, pose_labels))
            pose_color_maps[camera_letter] = {pose_2d_id: 'grey' for pose_2d_id in pose_2d_ids}
        match_aliases = iter(list(string.ascii_uppercase[:num_matches]))
        match_colors = iter(sns.color_palette('husl', n_colors=num_matches))
        for (pose_2d_id_a, pose_2d_id_b), row in df.iterrows():
            if row['match']:
                old_label_a = pose_label_maps['a'][pose_2d_id_a]
                old_label_b = pose_label_maps['b'][pose_2d_id_b]
                pose_label_maps['a'][pose_2d_id_a] = '{} ({})'.format(
                    old_label_a,
                    old_label_b
                )
                pose_label_maps['b'][pose_2d_id_b] = '{} ({})'.format(
                    old_label_b,
                    old_label_a
                )
                if generate_match_aliases:
                    match_alias = next(match_aliases)
                    pose_label_maps['a'][pose_2d_id_a] = match_alias
                    pose_label_maps['b'][pose_2d_id_b] = match_alias
                pose_color = next(match_colors)
                pose_color_maps['a'][pose_2d_id_a] = pose_color
                pose_color_maps['b'][pose_2d_id_b] = pose_color
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
            camera_name = honeycomb_io.fetch_camera_names([camera_id])[camera_id]
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
    pose_2d_ids = df.index.tolist()
    if pose_label_map is None:
        if 'track_label_2d' in df.columns:
            pose_labels = df['track_label_2d'].values.tolist()
        elif len(pose_2d_ids) <= 26:
            pose_labels = string.ascii_uppercase[:len(pose_2d_ids)]
        else:
            pose_labels = range(len(pose_2d_ids))
        pose_label_map = dict(zip(pose_2d_ids, pose_labels))
    if pose_color_map is None:
        pose_colors = sns.color_palette('husl', n_colors=len(pose_2d_ids))
        pose_color_map = dict(zip(pose_2d_ids, pose_colors))
    if draw_keypoint_connectors:
        if keypoint_connectors is None:
            pose_model = honeycomb_io.fetch_pose_model(
                pose_2d_id=pose_2d_ids[0]
            )
            keypoint_connectors = pose_model.get('keypoint_connectors')
    for pose_2d_id, row in df.iterrows():
        draw_pose_2d_opencv(
            row['keypoint_coordinates_2d'],
            draw_keypoint_connectors=draw_keypoint_connectors,
            keypoint_connectors=keypoint_connectors,
            pose_label=pose_label_map[pose_2d_id],
            pose_color=pose_color_map[pose_2d_id],
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
        pose_model = honeycomb_io.fetch_pose_model_by_pose_model_id(
            pose_model_id
        )
        if keypoint_connectors is None:
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if camera_names is None:
        camera_names = honeycomb_io.fetch_camera_names(
            camera_ids
        )
    if camera_calibrations is None:
        camera_calibrations = honeycomb_io.fetch_camera_calibrations(
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
        pose_model = honeycomb_io.fetch_pose_model_by_pose_model_id(
            pose_model_id
        )
        if keypoint_connectors is None:
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if camera_names is None:
        camera_names = honeycomb_io.fetch_camera_names(
            camera_ids
        )
    if camera_calibrations is None:
        camera_calibrations = honeycomb_io.fetch_camera_calibrations(
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
            for pose_3d_id, row in poses_3d_df[poses_3d_df['timestamp'] == selected_timestamp].iterrows():
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
        camera_calibrations = honeycomb_io.fetch_camera_calibrations(
            camera_ids=[camera_id_a, camera_id_b],
            start=timestamp.to_pydatetime(),
            end=timestamp.to_pydatetime()
        )
    if camera_names is None:
        camera_names = honeycomb_io.fetch_camera_names(
            camera_ids=[camera_id_a, camera_id_b]
        )
    fig_suptitle = timestamp.strftime(plot_title_datetime_format)
    save_filename = '{}_{}_{}.{}'.format(
        filename_prefix,
        slugify.slugify(pose_pair['pose_2d_id_a']),
        slugify.slugify(pose_pair['pose_2d_id_b']),
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
        extraction_columns = ['pose_2d_id_' + camera_letter, 'timestamp']
        extraction_columns.extend([single_camera_column + '_' + camera_letter for single_camera_column in single_camera_columns])
        target_columns = ['pose_2d_id', 'timestamp']
        target_columns.extend(single_camera_columns)
        column_map = dict(zip(extraction_columns, target_columns))
        df_single_camera = df.reindex(columns=extraction_columns)
        df_single_camera.rename(columns=column_map, inplace=True)
        df_single_camera.drop_duplicates(subset='pose_2d_id', inplace=True)
        df_single_camera.set_index('pose_2d_id', inplace=True)
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
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream)
    if delete_input_videos:
        for input_video_path in input_videos_path_list:
            os.remove(input_video_path)
    os.remove(temp_file_list_path)
