import process_pose_data.local_io
import process_pose_data.analyze
import multiprocessing
import functools
import logging
import datetime

logger = logging.getLogger(__name__)

def reconstruct_poses_3d_alphapose_local_by_time_segment(
    start,
    end,
    base_dir,
    environment_id,
    room_x_limits,
    room_y_limits,
    parallel=False,
    num_parallel_processes=None,
    input_file_name='alphapose-results.json',
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
    notebook=False
):
    time_segment_start_list = process_pose_data.local_io.generate_time_segment_start_list(
        start=start,
        end=end
    )
    logger.info('Reconstructing 3D poses for {} time segments: {} to {}'.format(
        len(time_segment_start_list),
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
        input_file_name=input_file_name,
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

def reconstruct_poses_3d_alphapose_local_time_segment(
    time_segment_start,
    base_dir,
    environment_id,
    input_file_name='alphapose-results.json',
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
        file_name=input_file_name
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
