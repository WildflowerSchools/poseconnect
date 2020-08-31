import process_pose_data.local_io
import process_pose_data.analyze
import click
import multiprocessing
import functools
import logging
import datetime
import time

logger = logging.getLogger(__name__)

@click.command()
@click.option('--start', required=True, type=click.DateTime(), help='Start of the time window to analyze in UTC')
@click.option('--end', required=True, type=click.DateTime(), help='End of the time window to analyze in UTC')
@click.option('--base-dir', required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), help='Base directory containing pose data tree')
@click.option('--environment-id', required=True, help='Honeycomb environment ID to analyze')
@click.option('--camera-assignment-id', 'camera_assignment_ids', required=True, multiple=True, help='Honeycomb camera assignment IDs contained in pose data (invoke once for each camera)')
@click.option('--room-x-limits', required=True, nargs=2, type=float, help='Spatial limits of room in first coordinate (e.g., -5.0 5.0)')
@click.option('--room-y-limits', required=True, nargs=2, type=float, help='Spatial limits of room in second coordinate (e.g., -5.0 5.0)')
@click.option('--pose-model-id', required=True, help='Honeycomb pose model ID for poses in data')
@click.option('--parallel/-no-parallel', default=False, help='Turn parallel processing on/off [default:  off]')
@click.option('--num-parallel-processes', type=int, help='Number of parallel processes to launch [default:  number of cores - 1]')
@click.option('--poses-2d-file-name', default='alphapose-results.json', show_default=True, help='File name for 2D pose data in each directory')
@click.option('--poses-3d-directory-name', default='poses_3d', show_default=True, help='Name of directory containing 3D pose data (just below environment ID level)')
@click.option('--poses-3d-file-name', default='poses_3d.pkl', show_default=True, help='File name for 3D pose data in each directory')
@click.option('--uri', help='Honeycomb URI (defaults to value of HONEYCOMB_URI environment variable)')
@click.option('--token-uri', help='Honeycomb token URI (defaults to value of HONEYCOMB_TOKEN_URI environment variable)')
@click.option('--audience', help='Honeycomb audience (defaults to value of HONEYCOMB_AUDIENCE environment variable)')
@click.option('--client-id', help='Honeycomb client ID (defaults to value of HONEYCOMB_CLIENT_ID environment variable)')
@click.option('--client-secret', help='Honeycomb client secret (defaults to value of HONEYCOMB_CLIENT_SECRET environment variable)')
@click.option('--min-keypoint-quality', type=float, help='Minimum keypoint quality for 2D pose keypoint to be included in analysis [default:  none]')
@click.option('--min-num-keypoints', type=int, help='Minimum number of valid keypoints (after keypoint quality filter) for 2D pose to be included in analysis [default:  none]')
@click.option('--min-pose-quality', type=float, help='Minimum pose quality for 2D pose to be included in analysis [default:  none]')
@click.option('--min-pose-pair-score', type=float, help='Minimum pose pair score for 2D pose pair to be included in analysis [default:  none]')
@click.option('--max-pose-pair-score', type=float, default=25.0, show_default=True, help='Maximum pose pair score for 2D pose pair to be included in analysis')
@click.option('--pose-pair-score-distance-method', default='pixels', show_default=True, help='Method for measuring reprojected keypoint distance in calculating pose pair score')
@click.option('--pose-pair-score-pixel-distance-scale', default=5.0, show_default=True, help='Pixel distance scale for \'probability\' distance method')
@click.option('--pose-pair-score-summary-method', default='rms', show_default=True, help='Method for summarizing distance over keypoints for pose pair score')
@click.option('--pose-3d-graph-initial-edge-threshold', type=int, default=2, show_default=True, help='Initial k value for defining 3D pose subgraphs')
@click.option('--pose-3d-graph-max-dispersion', type=float, default=0.20, show_default=True, help='Maximum spatial dispersion before k value is increased for a 3D pose subgraph')
@click.option('--include-track-labels/--no-track-labels', default=False, help='Include/don\'t include list of 2D pose track labels in 3D poses [default:  don\'t include]')
@click.option('--progress-bar/--no-progress-bar', default=False, help='Turn on/off progress bar [default:  off]')
@click.option('--log-level', help='Log level (e.g., warning, info, debug, etc.)')
def reconstruct_poses_3d_alphapose_local_by_time_segment(
    start,
    end,
    base_dir,
    environment_id,
    room_x_limits,
    room_y_limits,
    parallel=False,
    num_parallel_processes=None,
    poses_2d_file_name='alphapose-results.json',
    poses_3d_directory_name='poses_3d',
    poses_3d_file_name='poses_3d.pkl',
    camera_assignment_ids=None,
    camera_device_id_lookup=None,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
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
    pose_3d_graph_initial_edge_threshold=2,
    pose_3d_graph_max_dispersion=0.20,
    include_track_labels=False,
    progress_bar=False,
    notebook=False,
    log_level=None
):
    """
    Fetches 2D pose data (in AlphaPose format) from local drive, reconstructs 3D
    poses, and write results back to local drive.

    Structure of local data directories for 2D poses is assumed to be:
    BASE_DIR/ENVIRONMENT_ID/CAMERA_ASSIGNMENT_ID/YYYY/MM/DD/HH-MM-SS/POSES_2D_FILE_NAME

    Structure of local data directories for 3D poses is assumed to be:
    BASE_DIR/ENVIRONMENT_ID/POSES_3D_DIRECTORY_NAME/YYYY/MM/DD/HH-MM-SS/POSES_3D_FILE_NAME
    """

    if log_level is not None:
        numeric_log_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_log_level, int):
            raise ValueError('Invalid log level: %s'.format(log_level))
        logging.basicConfig(level=numeric_log_level)
    if progress_bar and parallel and ~notebook:
        logger.warning('Progress bars may not display properly with parallel processing enabled outside of a notebook')
    if start.tzinfo is None:
        start=start.replace(tzinfo=datetime.timezone.utc)
    if end.tzinfo is None:
        end=end.replace(tzinfo=datetime.timezone.utc)
    time_segment_start_list = process_pose_data.local_io.generate_time_segment_start_list(
        start=start,
        end=end
    )
    num_time_segments = len(time_segment_start_list)
    num_minutes = (end - start).seconds/60
    logger.info('Reconstructing 3D poses for {} time segments spanning {:.3f} minutes: {} to {}'.format(
        num_time_segments,
        num_minutes,
        time_segment_start_list[0].isoformat(),
        time_segment_start_list[-1].isoformat()
    ))
    if camera_device_id_lookup is None:
        logger.info('Camera device ID lookup table not specified. Fetching camera device ID info from Honeycomb based on specified camera assignment IDs')
        if camera_assignment_ids is None:
            raise ValueError('Must specify a list of camera assignment IDs present in the 2D pose data or a camera device ID lookup table')
        camera_device_id_lookup = process_pose_data.honeycomb_io.fetch_camera_device_id_lookup(
            assignment_ids=camera_assignment_ids,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    camera_device_ids = list(camera_device_id_lookup.values())
    if camera_calibrations is None:
        logger.info('Camera calibration parameters not specified. Fetching camera calibration parameters based on specified camera device IDs and time span')
        camera_calibrations = process_pose_data.honeycomb_io.fetch_camera_calibrations(
            camera_ids=camera_device_ids,
            start=min(time_segment_start_list),
            end=max(time_segment_start_list),
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    if pose_3d_limits is None:
        logger.info('3D pose spatial limits not specified. Calculating default spatial limits based on specified room spatial limits and specified pose model')
        if pose_model_id is None:
            raise ValueError('Must specify the pose model ID or explicitly specify the 3D pose spatial limits')
        pose_model = process_pose_data.honeycomb_io.fetch_pose_model_by_pose_model_id(
            pose_model_id,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
        pose_model_name = pose_model.get('model_name')
        pose_3d_limits = process_pose_data.analyze.pose_3d_limits_by_pose_model(
            room_x_limits=room_x_limits,
            room_y_limits=room_y_limits,
            pose_model_name=pose_model_name
        )
    reconstruct_poses_3d_alphapose_local_time_segment_partial = functools.partial(
        reconstruct_poses_3d_alphapose_local_time_segment,
        base_dir=base_dir,
        environment_id=environment_id,
        poses_2d_file_name=poses_2d_file_name,
        poses_3d_directory_name=poses_3d_directory_name,
        poses_3d_file_name=poses_3d_file_name,
        camera_device_id_lookup=camera_device_id_lookup,
        client=None,
        uri=None,
        token_uri=None,
        audience=None,
        client_id=None,
        client_secret=None,
        pose_model_id=None,
        camera_calibrations=camera_calibrations,
        min_keypoint_quality=min_keypoint_quality,
        min_num_keypoints=min_num_keypoints,
        min_pose_quality=min_pose_quality,
        min_pose_pair_score=min_pose_pair_score,
        max_pose_pair_score=max_pose_pair_score,
        pose_pair_score_distance_method=pose_pair_score_distance_method,
        pose_pair_score_pixel_distance_scale=pose_pair_score_pixel_distance_scale,
        pose_pair_score_summary_method=pose_pair_score_summary_method,
        pose_3d_limits=pose_3d_limits,
        room_x_limits=room_x_limits,
        room_y_limits=room_y_limits,
        pose_3d_graph_initial_edge_threshold=pose_3d_graph_initial_edge_threshold,
        pose_3d_graph_max_dispersion=pose_3d_graph_max_dispersion,
        include_track_labels=include_track_labels,
        progress_bar=progress_bar,
        notebook=notebook
    )
    processing_start = time.time()
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
            poses_3d_df_list = p.map(reconstruct_poses_3d_alphapose_local_time_segment_partial, time_segment_start_list)
    else:
        poses_3d_df_list = list(map(reconstruct_poses_3d_alphapose_local_time_segment_partial, time_segment_start_list))
    processing_time = time.time() - processing_start
    logger.info('Processed {:.3f} minutes of 2D poses in {:.3f} minutes (ratio of {:.3f})'.format(
        num_minutes,
        processing_time/60,
        (processing_time/60)/num_minutes
    ))

def reconstruct_poses_3d_alphapose_local_time_segment(
    time_segment_start,
    base_dir,
    environment_id,
    poses_2d_file_name='alphapose-results.json',
    poses_3d_directory_name='poses_3d',
    poses_3d_file_name='poses_3d.pkl',
    camera_device_id_lookup=None,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
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
    logger.info('Processing 2D poses from local Alphapose output files for time segment starting at {}'.format(time_segment_start.isoformat()))
    logger.info('Fetching 2D pose data for time segment starting at {}'.format(time_segment_start.isoformat()))
    poses_2d_df_time_segment = process_pose_data.local_io.fetch_2d_pose_data_alphapose_local_time_segment(
        base_dir=base_dir,
        environment_id=environment_id,
        time_segment_start=time_segment_start,
        file_name=poses_2d_file_name
    )
    logger.info('Fetched 2D pose data for time segment starting at {}'.format(time_segment_start.isoformat()))
    logger.info('Converting camera assignment IDs to camera device IDs for time segment starting at {}'.format(time_segment_start.isoformat()))
    poses_2d_df_time_segment = process_pose_data.local_io.convert_assignment_ids_to_camera_device_ids(
        poses_2d_df=poses_2d_df_time_segment,
        camera_device_id_lookup=camera_device_id_lookup,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    logger.info('Converted camera assignment IDs to camera device IDs for time segment starting at {}'.format(time_segment_start.isoformat()))
    logger.info('Reconstructing 3D poses for time segment starting at {}'.format(time_segment_start.isoformat()))
    poses_3d_local_ids_df = process_pose_data.analyze.reconstruct_poses_3d(
        poses_2d_df=poses_2d_df_time_segment,
        pose_model_id=pose_model_id,
        camera_calibrations=camera_calibrations,
        min_keypoint_quality=min_keypoint_quality,
        min_num_keypoints=min_num_keypoints,
        min_pose_quality=min_pose_quality,
        min_pose_pair_score=min_pose_pair_score,
        max_pose_pair_score=max_pose_pair_score,
        pose_pair_score_distance_method=pose_pair_score_distance_method,
        pose_pair_score_pixel_distance_scale=pose_pair_score_pixel_distance_scale,
        pose_pair_score_summary_method=pose_pair_score_summary_method,
        pose_3d_limits=pose_3d_limits,
        room_x_limits=room_x_limits,
        room_y_limits=room_y_limits,
        pose_3d_graph_initial_edge_threshold=pose_3d_graph_initial_edge_threshold,
        pose_3d_graph_max_dispersion=pose_3d_graph_max_dispersion,
        include_track_labels=include_track_labels,
        progress_bar=progress_bar,
        notebook=notebook
    )
    logger.info('Reconstructed 3D poses for time segment starting at {}'.format(time_segment_start.isoformat()))
    logger.info('Writing 3D poses to disk for time segment starting at {}'.format(time_segment_start.isoformat()))
    process_pose_data.local_io.write_3d_pose_data_local_time_segment(
        poses_3d_df=poses_3d_local_ids_df,
        base_dir=base_dir,
        environment_id=environment_id,
        time_segment_start=time_segment_start,
        directory_name=poses_3d_directory_name,
        file_name=poses_3d_file_name
    )
