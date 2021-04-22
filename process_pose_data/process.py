import process_pose_data.local_io
import process_pose_data.reconstruct
import process_pose_data.track_poses
import process_pose_data.identify
import process_pose_data.overlay
import honeycomb_io
import video_io
import pandas as pd
import tqdm
from uuid import uuid4
import multiprocessing
import functools
import logging
import datetime
import time

logger = logging.getLogger(__name__)

def extract_poses_2d_alphapose_local_by_time_segment(
    start,
    end,
    base_dir,
    environment_id,
    alphapose_subdirectory='prepared',
    tree_structure='file-per-frame',
    poses_2d_file_name='alphapose-results.json',
    poses_2d_json_format='cmu',
    pose_processing_subdirectory='pose_processing',
    task_progress_bar=False,
    notebook=False
):
    """
    Fetches 2D pose data from local Alphapose output and writes back to local files.

    Input data is assumed to be organized according to standard Alphapose
    pipeline output (see documentation for that pipeline).

    Output data is organized into 10 second segments (mirroring source videos),
    saved as
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/pose_extraction_2d/ENVIRONMENT_ID/YYYY/MM/DD/HH-MM-SS/poses_2d_INFERENCE_ID.pkl\'

    Output metadata is saved as
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/pose_extraction_2d/ENVIRONMENT_ID/pose_extraction_2d_metadata_INFERENCE_ID.pkl\'

    Args:
        start (datetime): Start of period to be analyzed
        end (datetime): End of period to be analyzed
        base_dir: Base directory for local data (e.g., \'/data\')
        environment_id (str): Honeycomb environment ID for source environment
        alphapose_subdirectory (str): subdirectory (under base directory) for Alphapose output (default is \'prepared\')
        poses_2d_file_name: Filename for Alphapose data in each directory (default is \'alphapose-results.json\')
        poses_2d_json_format: Format of Alphapose results files (default is\'cmu\')
        pose_processing_subdirectory (str): subdirectory (under base directory) for all pose processing data (default is \'pose_processing\')
        task_progress_bar (bool): Boolean indicating whether script should display a progress bar (default is False)
        notebook (bool): Boolean indicating whether script is being run in a Jupyter notebook (for progress bar display) (default is False)

    Returns:
        (str) Locally-generated inference ID for this run (identifies output data)
    """
    if start.tzinfo is None:
        logger.info('Specified start is timezone-naive. Assuming UTC')
        start=start.replace(tzinfo=datetime.timezone.utc)
    if end.tzinfo is None:
        logger.info('Specified end is timezone-naive. Assuming UTC')
        end=end.replace(tzinfo=datetime.timezone.utc)
    logger.info('Extracting 2D poses from local Alphapose output. Base directory: {}. Alphapose data subdirectory: {}. Pose processing data subdirectory: {}. Environment ID: {}. Start: {}. End: {}'.format(
        base_dir,
        alphapose_subdirectory,
        pose_processing_subdirectory,
        environment_id,
        start,
        end
    ))
    logger.info('Generating metadata')
    pose_extraction_2d_metadata = generate_metadata(
        environment_id=environment_id,
        pipeline_stage='pose_extraction_2d',
        parameters={
            'start': start,
            'end': end
        }
    )
    inference_id = pose_extraction_2d_metadata.get('inference_id')
    logger.info('Writing inference metadata to local file')
    process_pose_data.local_io.write_data_local(
        data_object=pose_extraction_2d_metadata,
        base_dir=base_dir,
        pipeline_stage='pose_extraction_2d',
        environment_id=environment_id,
        filename_stem='pose_extraction_2d_metadata',
        inference_id=pose_extraction_2d_metadata['inference_id'],
        time_segment_start=None,
        object_type='dict',
        append=False,
        sort_field=None,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    logger.info('Generating list of time segments')
    time_segment_start_list = process_pose_data.local_io.generate_time_segment_start_list(
        start=start,
        end=end
    )
    num_time_segments = len(time_segment_start_list)
    num_minutes = (end - start).total_seconds()/60
    logger.info('Extracting 2D poses for {} time segments spanning {:.3f} minutes: {} to {}'.format(
        num_time_segments,
        num_minutes,
        time_segment_start_list[0].isoformat(),
        time_segment_start_list[-1].isoformat()
    ))
    processing_start = time.time()
    if task_progress_bar:
        if notebook:
            time_segment_start_iterator = tqdm.notebook.tqdm(time_segment_start_list)
        else:
            time_segment_start_iterator = tqdm.tqdm(time_segment_start_list)
    else:
        time_segment_start_iterator = time_segment_start_list
    for time_segment_start in time_segment_start_iterator:
        poses_2d_df_time_segment = process_pose_data.local_io.fetch_2d_pose_data_alphapose_local_time_segment(
            base_dir=base_dir,
            environment_id=environment_id,
            time_segment_start=time_segment_start,
            alphapose_subdirectory=alphapose_subdirectory,
            tree_structure=tree_structure,
            filename=poses_2d_file_name,
            json_format=poses_2d_json_format
        )
        process_pose_data.local_io.write_data_local(
            data_object=poses_2d_df_time_segment,
            base_dir=base_dir,
            pipeline_stage='pose_extraction_2d',
            environment_id=environment_id,
            filename_stem='poses_2d',
            inference_id=inference_id,
            time_segment_start=time_segment_start,
            object_type='dataframe',
            append=False,
            sort_field=None,
            pose_processing_subdirectory=pose_processing_subdirectory
        )
    processing_time = time.time() - processing_start
    logger.info('Extracted {:.3f} minutes of 2D poses in {:.3f} minutes (ratio of {:.3f})'.format(
        num_minutes,
        processing_time/60,
        (processing_time/60)/num_minutes
    ))
    return inference_id

def reconstruct_poses_3d_local_by_time_segment(
    base_dir,
    environment_id,
    pose_extraction_2d_inference_id,
    pose_model_id,
    room_x_limits,
    room_y_limits,
    start=None,
    end=None,
    camera_assignment_ids=None,
    camera_device_id_lookup=None,
    camera_calibrations=None,
    coordinate_space_id=None,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    pose_processing_subdirectory='pose_processing',
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
    parallel=False,
    num_parallel_processes=None,
    task_progress_bar=False,
    segment_progress_bar=False,
    notebook=False
):
    """
    Fetches 2D pose data from local files, reconstructs 3D poses, and writes output back to local files.

    If camera information is not specified, script pulls camera data from
    Honeycomb based on environment ID, start time, and end time.

    Options for pose pair score distance method are \'pixels\' (simple 2D distance
    measured in pixels) or \'probability\' (likelihood of that distance assuming
    2D Gaussian reprojection error).

    Options for pose pair score summary method are \'rms\' (root mean square of
    distances across keypoints) or \'sum\' (sum of distances across keypoints).

    3D pose limits are an array with shape (2, NUM_KEYPOINTS, 3) specifying the
    minimum and maximum possible coordinate values for each type of keypoint
    (for filtering 3D poses). If these limits are not specified, script
    calculates default limits based on room x and y limits and pose model.

    Candidate 3D poses are validated and grouped into people using an adaptive
    strategy. After initial pose filtering, the algorithm forms a graph with 2D
    poses as nodes and candidate 3D poses as edges. This graph is then split
    into k-edge-connected subgraphs (people), starting with the specified
    initial k (i.e., if k > 1, each person must be confirmed by matches across
    multiple cameras). If the spatial dispersion of the 3D poses for any
    subgraph (person) exceeds the specified threshold (suggesting that multiple
    people are being conflated), k is increased for that subgraph, effectively
    splitting the subgraph. This process repeats until all subgraphs (people)
    fall below the maximum spatial dispersion (a valid person) or below minimum
    k (poses are rejected).

    Input data is assumed to be organized as specified by
    extract_poses_2d_alphapose_local_by_time_segment().

    Output data is organized into 10 second segments (mirroring source videos),
    saved as
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/pose_reconstruction_3d/ENVIRONMENT_ID/YYYY/MM/DD/HH-MM-SS/poses_3d_INFERENCE_ID.pkl\'

    Output metadata is saved as
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/pose_reconstruction_3d/ENVIRONMENT_ID/pose_reconstruction_3d_metadata_INFERENCE_ID.pkl\'

    Args:
        base_dir: Base directory for local data (e.g., \'/data\')
        environment_id (str): Honeycomb environment ID for source environment
        pose_extraction_2d_inference_id (str): Inference ID for source data
        pose_model_id (str): Honeycomb pose model ID for pose model that defines 2D/3D pose data structure
        room_x_limits (sequence of float): Boundaries of room in x direction in [MIN, MAX] format (for filtering 3D poses)
        room_y_limits (sequence of float): Boundaries of room in y direction in [MIN, MAX] format (for filtering 3D poses)
        start (datetime): Start of period within source data to be analyzed (default is None)
        end (datetime): End of period within source data to be analyzed (default is None)
        camera_assignment_ids (sequence of str): List of camera assignment IDs to analyze (default is None)
        camera_device_id_lookup (dict): Dict in format {ASSSIGNMENT_ID: DEVICE ID} (default is None)
        camera_calibrations (dict): Dict in format {DEVICE_ID: CAMERA_CALIBRATION_DATA} (default is None)
        coordinate_space_id (dict): Coordinate space ID of extrinsic camera calibrations (and therefore 3D pose output) (default is None)
        client (MinimalHoneycombClient): Honeycomb client (otherwise generates one) (default is None)
        uri (str): Honeycomb URI (otherwise falls back on default strategy of MinimalHoneycombClient) (default is None)
        token_uri (str): Honeycomb token URI (otherwise falls back on default strategy of MinimalHoneycombClient) (default is None)
        audience (str): Honeycomb audience (otherwise falls back on default strategy of MinimalHoneycombClient) (default is None)
        client_id (str): Honeycomb client ID (otherwise falls back on default strategy of MinimalHoneycombClient) (default is None)
        client_secret (str): Honeycomb client secret (otherwise falls back on default strategy of MinimalHoneycombClient) (default is None)
        pose_processing_subdirectory (str): subdirectory (under base directory) for all pose processing data (default is \'pose_processing\')
        min_keypoint_quality (float): Minimum keypoint quality for keypoint to be included (default is None)
        min_num_keypoints (float): Mininum number of keypoints (after keypoint quality filter) for 2D pose to be included (default is None)
        min_pose_quality=None (float): Minimum pose quality for 2D pose to be included (default is None)
        min_pose_pair_score (float): Minimum pose pair score for pose pair to be included (default is None)
        max_pose_pair_score (float): Maximum pose pair for pose pair to be included (default is 25.0)
        pose_pair_score_distance_method (str): Method for calculating distance between original and reprojected pose keypoints (default is \'pixels\')
        pose_pair_score_pixel_distance_scale (float): Pixel distance scale for \'probability\' method (default is 5.0)
        pose_pair_score_summary_method (str): Method for summarizing reprojected keypoint distance over pose (default is \'rms\')
        pose_3d_limits (array): Spatial limits for each type of pose keypoint (for filtering candidate 3D poses) (default is None)
        pose_3d_graph_initial_edge_threshold (int): Minimum number of pose pairs in pose (edges in graph) (default is 2)
        pose_3d_graph_max_dispersion (float): Keypoint dispersion threshold for increasing required number of edges (default is 0.20)
        include_track_labels (bool): Boolean indicating whether to include source 2D track labels in 3D pose data (default is False)
        parallel (bool): Boolean indicating whether to use multiple parallel processes (one for each time segment) (default is False)
        num_parallel_processes (int): Number of parallel processes in pool (otherwise defaults to number of cores - 1) (default is None)
        task_progress_bar (bool): Boolean indicating whether script should display an overall progress bar (default is False)
        segment_progress_bar (bool): Boolean indicating whether script should display a progress bar for each time segment (default is False)
        notebook (bool): Boolean indicating whether script is being run in a Jupyter notebook (for progress bar display) (default is False)

    Returns:
        (str) Locally-generated inference ID for this run (identifies output data)
    """
    pose_extraction_2d_metadata = process_pose_data.local_io.fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_extraction_2d',
        environment_id=environment_id,
        filename_stem='pose_extraction_2d_metadata',
        inference_ids=pose_extraction_2d_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=None,
        object_type='dict',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    if start is None:
        start = pose_extraction_2d_metadata['parameters']['start']
    if end is None:
        end = pose_extraction_2d_metadata['parameters']['end']
    if start.tzinfo is None:
        logger.info('Specified start is timezone-naive. Assuming UTC')
        start=start.replace(tzinfo=datetime.timezone.utc)
    if end.tzinfo is None:
        logger.info('Specified end is timezone-naive. Assuming UTC')
        end=end.replace(tzinfo=datetime.timezone.utc)
    logger.info('Reconstructing 3D poses from local 2D pose data. Base directory: {}. Pose processing data subdirectory: {}. Environment ID: {}. Start: {}. End: {}'.format(
        base_dir,
        pose_processing_subdirectory,
        environment_id,
        start,
        end
    ))
    logger.info('Generating metadata')
    if camera_assignment_ids is None:
        logger.info('Camera assignment IDs not specified. Fetching camera assignment IDs from Honeycomb based on environmen and time span')
        camera_assignment_ids = honeycomb_io.fetch_camera_assignment_ids_from_environment(
            start=start,
            end=end,
            environment_id=environment_id,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    if camera_device_id_lookup is None:
        logger.info('Camera device ID lookup table not specified. Fetching camera device ID info from Honeycomb based on camera assignment IDs')
        camera_device_id_lookup = honeycomb_io.fetch_camera_device_id_lookup(
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
        logger.info('Camera calibration parameters not specified. Fetching camera calibration parameters from Honeycomb based on camera device IDs and time span')
        camera_calibrations = honeycomb_io.fetch_camera_calibrations(
            camera_ids=camera_device_ids,
            start=start,
            end=end,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    if coordinate_space_id is None:
        coordinate_space_id = extract_coordinate_space_id_from_camera_calibrations(camera_calibrations)
    if pose_3d_limits is None:
        logger.info('3D pose spatial limits not specified. Generating default spatial limits based on specified room spatial limits and specified pose model')
        pose_3d_limits = generate_pose_3d_limits(
            pose_model_id=pose_model_id,
            room_x_limits=room_x_limits,
            room_y_limits=room_y_limits,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    pose_reconstruction_3d_metadata = generate_metadata(
        environment_id=environment_id,
        pipeline_stage='pose_reconstruction_3d',
        parameters={
            'pose_extraction_2d_inference_id': pose_extraction_2d_inference_id,
            'start': start,
            'end': end,
            'pose_model_id': pose_model_id,
            'room_x_limits': room_x_limits,
            'room_y_limits': room_y_limits,
            'camera_assignment_ids': camera_assignment_ids,
            'camera_device_id_lookup': camera_device_id_lookup,
            'camera_device_ids': camera_device_ids,
            'camera_calibrations': camera_calibrations,
            'coordinate_space_id': coordinate_space_id,
            'min_keypoint_quality': min_keypoint_quality,
            'min_num_keypoints': min_num_keypoints,
            'min_pose_quality': min_pose_quality,
            'min_pose_pair_score': min_pose_pair_score,
            'max_pose_pair_score': max_pose_pair_score,
            'pose_pair_score_distance_method': pose_pair_score_distance_method,
            'pose_pair_score_pixel_distance_scale': pose_pair_score_pixel_distance_scale,
            'pose_pair_score_summary_method': pose_pair_score_summary_method,
            'pose_3d_limits': pose_3d_limits,
            'pose_3d_graph_initial_edge_threshold': pose_3d_graph_initial_edge_threshold,
            'pose_3d_graph_max_dispersion': pose_3d_graph_max_dispersion,
            'include_track_labels': include_track_labels

        }
    )
    inference_id = pose_reconstruction_3d_metadata.get('inference_id')
    logger.info('Writing inference metadata to local file')
    process_pose_data.local_io.write_data_local(
        data_object=pose_reconstruction_3d_metadata,
        base_dir=base_dir,
        pipeline_stage='pose_reconstruction_3d',
        environment_id=environment_id,
        filename_stem='pose_reconstruction_3d_metadata',
        inference_id=pose_reconstruction_3d_metadata['inference_id'],
        time_segment_start=None,
        object_type='dict',
        append=False,
        sort_field=None,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    logger.info('Generating list of time segments')
    time_segment_start_list = process_pose_data.local_io.generate_time_segment_start_list(
        start=start,
        end=end
    )
    num_time_segments = len(time_segment_start_list)
    num_minutes = (end - start).total_seconds()/60
    logger.info('Reconstructing 3D poses for {} time segments spanning {:.3f} minutes: {} to {}'.format(
        num_time_segments,
        num_minutes,
        time_segment_start_list[0].isoformat(),
        time_segment_start_list[-1].isoformat()
    ))
    reconstruct_poses_3d_alphapose_local_time_segment_partial = functools.partial(
        reconstruct_poses_3d_alphapose_local_time_segment,
        base_dir=base_dir,
        environment_id=environment_id,
        pose_extraction_2d_inference_id=pose_extraction_2d_inference_id,
        pose_reconstruction_3d_inference_id=inference_id,
        pose_processing_subdirectory=pose_processing_subdirectory,
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
        progress_bar=segment_progress_bar,
        notebook=notebook
    )
    if (task_progress_bar or segment_progress_bar) and parallel and not notebook:
        logger.warning('Progress bars may not display properly with parallel processing enabled outside of a notebook')
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
            if task_progress_bar:
                if notebook:
                    list(tqdm.notebook.tqdm(
                        p.imap_unordered(
                            reconstruct_poses_3d_alphapose_local_time_segment_partial,
                            time_segment_start_list
                        ),
                        total=len(time_segment_start_list)
                    ))
                else:
                    list(tqdm.tqdm(
                        p.imap_unordered(
                            reconstruct_poses_3d_alphapose_local_time_segment_partial,
                            time_segment_start_list
                        ),
                        total=len(time_segment_start_list)
                    ))
            else:
                list(
                    p.imap_unordered(
                        reconstruct_poses_3d_alphapose_local_time_segment_partial,
                        time_segment_start_list
                    )
                )
    else:
        if task_progress_bar:
            if notebook:
                list(map(reconstruct_poses_3d_alphapose_local_time_segment_partial, tqdm.notebook.tqdm(time_segment_start_list)))
            else:
                list(map(reconstruct_poses_3d_alphapose_local_time_segment_partial, tqdm.tqdm(time_segment_start_list)))
        else:
            list(map(reconstruct_poses_3d_alphapose_local_time_segment_partial, time_segment_start_list))
    processing_time = time.time() - processing_start
    logger.info('Processed {:.3f} minutes of 2D poses in {:.3f} minutes (ratio of {:.3f})'.format(
        num_minutes,
        processing_time/60,
        (processing_time/60)/num_minutes
    ))
    return inference_id

def reconstruct_poses_3d_alphapose_local_time_segment(
    time_segment_start,
    base_dir,
    environment_id,
    pose_extraction_2d_inference_id,
    pose_reconstruction_3d_inference_id,
    pose_processing_subdirectory='pose_processing',
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
    poses_2d_df_time_segment = process_pose_data.local_io.fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_extraction_2d',
        environment_id=environment_id,
        filename_stem='poses_2d',
        inference_ids=pose_extraction_2d_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=time_segment_start,
        object_type='dataframe',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    if len(poses_2d_df_time_segment) == 0:
        logger.info('No 2D poses found for time segment starting at %s', time_segment_start.isoformat())
        return
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
    poses_3d_df = process_pose_data.reconstruct.reconstruct_poses_3d(
        poses_2d_df=poses_2d_df_time_segment,
        pose_2d_id_column_name='pose_2d_id',
        pose_2d_ids_column_name='pose_2d_ids',
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
    process_pose_data.local_io.write_data_local(
        data_object=poses_3d_df,
        base_dir=base_dir,
        pipeline_stage='pose_reconstruction_3d',
        environment_id=environment_id,
        filename_stem='poses_3d',
        inference_id=pose_reconstruction_3d_inference_id,
        time_segment_start=time_segment_start,
        object_type='dataframe',
        append=False,
        sort_field=None,
        pose_processing_subdirectory=pose_processing_subdirectory
    )

def generate_pose_tracks_3d_local_by_time_segment(
    base_dir,
    environment_id,
    pose_reconstruction_3d_inference_id,
    start=None,
    end=None,
    max_match_distance=1.0,
    max_iterations_since_last_match=20,
    centroid_position_initial_sd=1.0,
    centroid_velocity_initial_sd=1.0,
    reference_delta_t_seconds=1.0,
    reference_velocity_drift=0.30,
    position_observation_sd=0.5,
    num_poses_per_track_min=11,
    pose_processing_subdirectory='pose_processing',
    task_progress_bar=False,
    notebook=False
):
    """
    Fetches 3D pose data from local files, assembles them into pose tracks, and writes output back to local files.

    Input data is assumed to be organized as specified by output of
    reconstruct_poses_3d_local_by_time_segment().

    Output data is saved as
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/pose_tracking_3d/ENVIRONMENT_ID/pose_tracks_3d_INFERENCE_ID.pkl\'

    Output metadata is saved as
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/pose_tracking_3d/ENVIRONMENT_ID/pose_tracking_3d_metadata_INFERENCE_ID.pkl\'

    Args:
        base_dir: Base directory for local data (e.g., \'/data\')
        environment_id (str): Honeycomb environment ID for source environment
        pose_reconstruction_3d_inference_id (str): Inference ID for source data
        start (datetime): Start of period within source data to be analyzed (default is None)
        end (datetime): End of period within source data to be analyzed (default is None)
        max_match_distance (float): Maximum distance between 3D pose and predicted pose track for pose to be added to track (default is 1.0)
        max_iterations_since_last_match (int): Maximum number of unmatched iterations before pose track is terminated (default is 20)
        centroid_position_initial_sd (float): Initial standard deviation for pose track centroid position (default is 1.0)
        centroid_velocity_initial_sd (float): Initial standard deviation for pose track centroid velocity (default is 1.0)
        reference_delta_t_seconds (float): Reference time period for specifying velocity drift (default is 1.0)
        reference_velocity_drift (float): Reference velocity drift (default is 0.30)
        position_observation_sd (float): Position observation error (reference is 0.5)
        num_poses_per_track_min (it): Mininum number of poses in a track (default is 11)
        pose_processing_subdirectory (str): subdirectory (under base directory) for all pose processing data (default is \'pose_processing\')
        task_progress_bar (bool): Boolean indicating whether script should display an overall progress bar (default is False)
        notebook (bool): Boolean indicating whether script is being run in a Jupyter notebook (for progress bar display) (default is False)

    Returns:
        (str) Locally-generated inference ID for this run (identifies output data)
    """
    pose_reconstruction_3d_metadata = process_pose_data.local_io.fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_reconstruction_3d',
        environment_id=environment_id,
        filename_stem='pose_reconstruction_3d_metadata',
        inference_ids=pose_reconstruction_3d_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=None,
        object_type='dict',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    if start is None:
        start = pose_reconstruction_3d_metadata['parameters']['start']
    if end is None:
        end = pose_reconstruction_3d_metadata['parameters']['end']
    if start.tzinfo is None:
        logger.info('Specified start is timezone-naive. Assuming UTC')
        start=start.replace(tzinfo=datetime.timezone.utc)
    if end.tzinfo is None:
        logger.info('Specified end is timezone-naive. Assuming UTC')
        end=end.replace(tzinfo=datetime.timezone.utc)
    logger.info('Generating 3D pose tracks from local 3D pose data. Base directory: {}. Pose processing data subdirectory: {}. Environment ID: {}. Start: {}. End: {}'.format(
        base_dir,
        pose_processing_subdirectory,
        environment_id,
        start,
        end
    ))
    logger.info('Generating metadata')
    pose_tracking_3d_metadata = generate_metadata(
        environment_id=environment_id,
        pipeline_stage='pose_tracking_3d',
        parameters={
            'pose_reconstruction_3d_inference_id': pose_reconstruction_3d_inference_id,
            'start': start,
            'end': end,
            'max_match_distance': max_match_distance,
            'max_iterations_since_last_match': max_iterations_since_last_match,
            'centroid_position_initial_sd': centroid_position_initial_sd,
            'centroid_velocity_initial_sd': None,
            'reference_delta_t_seconds': reference_delta_t_seconds,
            'reference_velocity_drift': reference_velocity_drift,
            'position_observation_sd': position_observation_sd,
            'num_poses_per_track_min': num_poses_per_track_min
        }
    )
    pose_tracking_3d_inference_id = pose_tracking_3d_metadata.get('inference_id')
    logger.info('Writing inference metadata to local file')
    process_pose_data.local_io.write_data_local(
        data_object=pose_tracking_3d_metadata,
        base_dir=base_dir,
        pipeline_stage='pose_tracking_3d',
        environment_id=environment_id,
        filename_stem='pose_tracking_3d_metadata',
        inference_id=pose_tracking_3d_metadata['inference_id'],
        time_segment_start=None,
        object_type='dict',
        append=False,
        sort_field=None,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    logger.info('Generating list of time segments')
    time_segment_start_list = process_pose_data.local_io.generate_time_segment_start_list(
        start=start,
        end=end
    )
    num_time_segments = len(time_segment_start_list)
    num_minutes = (end - start).total_seconds()/60
    logger.info('Tracking 3D poses for {} time segments spanning {:.3f} minutes: {} to {}'.format(
        num_time_segments,
        num_minutes,
        time_segment_start_list[0].isoformat(),
        time_segment_start_list[-1].isoformat()
    ))
    processing_start = time.time()
    pose_tracks_3d = None
    if task_progress_bar:
        if notebook:
            time_segment_start_iterator = tqdm.notebook.tqdm(time_segment_start_list)
        else:
            time_segment_start_iterator = tqdm.tqdm(time_segment_start_list)
    else:
        time_segment_start_iterator = time_segment_start_list
    for time_segment_start in time_segment_start_iterator:
        poses_3d_df = process_pose_data.local_io.fetch_data_local(
            base_dir=base_dir,
            pipeline_stage='pose_reconstruction_3d',
            environment_id=environment_id,
            filename_stem='poses_3d',
            inference_ids=pose_reconstruction_3d_inference_id,
            data_ids=None,
            sort_field=None,
            time_segment_start=time_segment_start,
            object_type='dataframe',
            pose_processing_subdirectory='pose_processing'
        )
        if len(poses_3d_df) == 0:
            continue
        pose_tracks_3d =  process_pose_data.track_poses.update_pose_tracks_3d(
            poses_3d_df=poses_3d_df,
            pose_tracks_3d=pose_tracks_3d,
            max_match_distance=max_match_distance,
            max_iterations_since_last_match=max_iterations_since_last_match,
            centroid_position_initial_sd=centroid_position_initial_sd,
            centroid_velocity_initial_sd=centroid_velocity_initial_sd,
            reference_delta_t_seconds=reference_delta_t_seconds,
            reference_velocity_drift=reference_velocity_drift,
            position_observation_sd=position_observation_sd,
            progress_bar=False,
            notebook=False
        )
    if num_poses_per_track_min is not None:
        pose_tracks_3d.filter(
            num_poses_min=num_poses_per_track_min,
            inplace=True
        )
    process_pose_data.local_io.write_data_local(
        data_object=pose_tracks_3d.output(),
        base_dir=base_dir,
        pipeline_stage='pose_tracking_3d',
        environment_id=environment_id,
        filename_stem='pose_tracks_3d',
        inference_id=pose_tracking_3d_inference_id,
        time_segment_start=None,
        object_type='dict',
        append=False,
        sort_field=None,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    processing_time = time.time() - processing_start
    logger.info('Processed {:.3f} minutes of 3D poses in {:.3f} minutes (ratio of {:.3f})'.format(
        num_minutes,
        processing_time/60,
        (processing_time/60)/num_minutes
    ))
    return pose_tracking_3d_inference_id

def interpolate_pose_tracks_3d_local_by_pose_track(
    base_dir,
    environment_id,
    pose_tracking_3d_inference_id,
    pose_processing_subdirectory='pose_processing',
    task_progress_bar=False,
    notebook=False
):
    """
    Fetches 3D pose and pose track data from local files, interpolates to fill gaps in the tracks, and writes output back to local files.

    Input data is assumed to be organized as specified by output of
    reconstruct_poses_3d_local_by_time_segment() and
    generate_pose_tracks_3d_local_by_time_segment().

    The script looks up the inference ID for the 3D poses in the tracks by
    inspecting the metadata from the pose tracking run.

    Output data is saved as
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/pose_reconstruction_3d/ENVIRONMENT_ID/YYYY/MM/DD/HH-MM-SS/poses_3d_INFERENCE_ID.pkl\'
    (for new poses) and
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/pose_tracking_3d/ENVIRONMENT_ID/pose_tracks_3d_INFERENCE_ID.pkl\'
    (for new pose track data).

    Output metadata is saved as
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/pose_track_3d_interpolation/ENVIRONMENT_ID/pose_track_3d_interpolation_metadata_INFERENCE_ID.pkl\'

    Args:
        base_dir: Base directory for local data (e.g., \'/data\')
        environment_id (str): Honeycomb environment ID for source environment
        pose_tracking_3d_inference_id (str): Inference ID for source data
        pose_processing_subdirectory (str): subdirectory (under base directory) for all pose processing data (default is \'pose_processing\')
        task_progress_bar (bool): Boolean indicating whether script should display an overall progress bar (default is False)
        notebook (bool): Boolean indicating whether script is being run in a Jupyter notebook (for progress bar display) (default is False)

    Returns:
        (str) Locally-generated inference ID for this run (identifies output data)
    """
    pose_tracking_3d_metadata = process_pose_data.local_io.fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_tracking_3d',
        environment_id=environment_id,
        filename_stem='pose_tracking_3d_metadata',
        inference_ids=pose_tracking_3d_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=None,
        object_type='dict',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    start = pose_tracking_3d_metadata['parameters']['start']
    end = pose_tracking_3d_metadata['parameters']['end']
    pose_reconstruction_3d_inference_id = pose_tracking_3d_metadata['parameters']['pose_reconstruction_3d_inference_id']
    logger.info('Interpolating 3D pose tracks from local 3D pose track data and local 3D pose data. Base directory: {}. Pose processing data subdirectory: {}. Environment ID: {}.'.format(
        base_dir,
        pose_processing_subdirectory,
        environment_id
    ))
    logger.info('Generating metadata')
    pose_track_3d_interpolation_metadata = generate_metadata(
        environment_id=environment_id,
        pipeline_stage='pose_track_3d_interpolation',
        parameters={
            'pose_reconstruction_3d_inference_id': pose_reconstruction_3d_inference_id,
            'pose_tracking_3d_inference_id': pose_tracking_3d_inference_id,
            'start': start,
            'end': end
        }
    )
    pose_track_3d_interpolation_inference_id = pose_track_3d_interpolation_metadata.get('inference_id')
    logger.info('Writing inference metadata to local file')
    process_pose_data.local_io.write_data_local(
        data_object=pose_track_3d_interpolation_metadata,
        base_dir=base_dir,
        pipeline_stage='pose_track_3d_interpolation',
        environment_id=environment_id,
        filename_stem='pose_track_3d_interpolation_metadata',
        inference_id=pose_track_3d_interpolation_metadata['inference_id'],
        time_segment_start=None,
        object_type='dict',
        append=False,
        sort_field=None,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    pose_tracks_3d = process_pose_data.local_io.fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_tracking_3d',
        environment_id=environment_id,
        filename_stem='pose_tracks_3d',
        inference_ids=pose_tracking_3d_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=None,
        object_type='dict',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    num_pose_tracks = len(pose_tracks_3d)
    pose_tracks_start = min([pose_track_3d['start'] for pose_track_3d in pose_tracks_3d.values()])
    pose_tracks_end = max([pose_track_3d['end'] for pose_track_3d in pose_tracks_3d.values()])
    num_poses = sum([len(pose_track_3d['pose_3d_ids']) for pose_track_3d in pose_tracks_3d.values()])
    num_minutes = (pose_tracks_end - pose_tracks_start).total_seconds()/60
    logger.info('Interpolating {} 3D pose tracks spanning {} poses and {:.3f} minutes: {} to {}'.format(
        num_pose_tracks,
        num_poses,
        num_minutes,
        pose_tracks_start.isoformat(),
        pose_tracks_end.isoformat()
    ))
    processing_start = time.time()
    if task_progress_bar:
        if notebook:
            pose_track_iterator = tqdm.notebook.tqdm(pose_tracks_3d.items())
        else:
            pose_track_iterator = tqdm.tqdm(pose_tracks_3d.items())
    else:
        pose_track_iterator = pose_tracks_3d.items()
    pose_tracks_3d_new = dict()
    for pose_track_3d_id, pose_track_3d in pose_track_iterator:
        pose_track_start = pose_track_3d['start']
        pose_track_end = pose_track_3d['end']
        pose_3d_ids = pose_track_3d['pose_3d_ids']
        poses_3d_in_track_df = process_pose_data.local_io.fetch_data_local_by_time_segment(
            start=pose_track_start,
            end=pose_track_end,
            base_dir=base_dir,
            pipeline_stage='pose_reconstruction_3d',
            environment_id=environment_id,
            filename_stem='poses_3d',
            inference_ids=pose_reconstruction_3d_inference_id,
            data_ids=pose_3d_ids,
            sort_field=None,
            object_type='dataframe',
            pose_processing_subdirectory='pose_processing'
        )
        poses_3d_new_df = process_pose_data.interpolate_pose_track(poses_3d_in_track_df)
        if len(poses_3d_new_df) == 0:
            continue
        process_pose_data.local_io.write_data_local_by_time_segment(
            data_object=poses_3d_new_df,
            base_dir=base_dir,
            pipeline_stage='pose_reconstruction_3d',
            environment_id=environment_id,
            filename_stem='poses_3d',
            inference_id=pose_track_3d_interpolation_inference_id,
            object_type='dataframe',
            append=True,
            sort_field=None,
            pose_processing_subdirectory=pose_processing_subdirectory
        )
        pose_tracks_3d_new[pose_track_3d_id] = {
            'start': pd.to_datetime(poses_3d_new_df['timestamp'].min()).to_pydatetime(),
            'end': pd.to_datetime(poses_3d_new_df['timestamp'].max()).to_pydatetime(),
            'pose_3d_ids': poses_3d_new_df.index.tolist()
        }
    process_pose_data.local_io.write_data_local(
        data_object=pose_tracks_3d_new,
        base_dir=base_dir,
        pipeline_stage='pose_tracking_3d',
        environment_id=environment_id,
        filename_stem='pose_tracks_3d',
        inference_id=pose_track_3d_interpolation_inference_id,
        time_segment_start=None,
        object_type='dict',
        append=False,
        sort_field=None,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    processing_time = time.time() - processing_start
    logger.info('Processed {} 3D pose tracks in {:.3f} minutes'.format(
        num_pose_tracks,
        processing_time/60
    ))
    return pose_track_3d_interpolation_inference_id

def download_position_data_by_datapoint(
    start,
    end,
    base_dir,
    environment_id,
    source_objects='position_objects',
    datapoint_timestamp_min=None,
    datapoint_timestamp_max=None,
    pose_processing_subdirectory='pose_processing',
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    task_progress_bar=False,
    notebook=False
):
    """
    Fetches UWB position data from Honeycomb and writes it back to local files.

    Determination of minimum and maximum datapoint timestamps for a given start
    and end time is tricky, because the timestamp on a UWB datapoint typically
    captures when the data in that datapoint begins but the duration of the data
    in that datapoint is less predictable (typically about 30 minutes). For this
    reason, the script asks the user to explicitly specify minimum and maximum
    datapoint timestamps rather than calculating them from the specified start and
    end times. A reasonable practice is to set the minimum datapoint timestamp
    to be about 40 minutes less than the start time and to set the maximum
    datapoint timestamp to be equal to the end time.

    Output data is organized into 10 second segments (mirroring videos) and
    saved as
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/download_position_data/ENVIRONMENT_ID/YYYY/MM/DD/HH-MM-SS/position_data_INFERENCE_ID.pkl\'.

    Output metadata is saved as
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/download_position_data/ENVIRONMENT_ID/download_position_data_metadata_INFERENCE_ID.pkl\'

    Args:
        datapoint_timestamp_min (datetime): Minimum UWB data datapoint timestamp to fetch
        datapoint_timestamp_max (datetime): Maximum UWB data datapoint timestamp to fetch
        start (datetime): Start of position data to fetch
        end (datetime): End of position data to fetch
        base_dir: Base directory for local data (e.g., \'/data\')
        environment_id (str): Honeycomb environment ID for source environment
        pose_processing_subdirectory (str): subdirectory (under base directory) for all pose processing data (default is \'pose_processing\')
        chunk_size (int): Maximum number of records to pull with Honeycomb request (default is 100)
        client (MinimalHoneycombClient): Honeycomb client (otherwise generates one) (default is None)
        uri (str): Honeycomb URI (otherwise falls back on default strategy of MinimalHoneycombClient) (default is None)
        token_uri (str): Honeycomb token URI (otherwise falls back on default strategy of MinimalHoneycombClient) (default is None)
        audience (str): Honeycomb audience (otherwise falls back on default strategy of MinimalHoneycombClient) (default is None)
        client_id (str): Honeycomb client ID (otherwise falls back on default strategy of MinimalHoneycombClient) (default is None)
        client_secret (str): Honeycomb client secret (otherwise falls back on default strategy of MinimalHoneycombClient) (default is None)
        task_progress_bar (bool): Boolean indicating whether script should display an overall progress bar (default is False)
        notebook (bool): Boolean indicating whether script is being run in a Jupyter notebook (for progress bar display) (default is False)

    Returns:
        (str) Locally-generated inference ID for this run (identifies output data)
    """
    if start.tzinfo is None:
        logger.info('Specified start is timezone-naive. Assuming UTC')
        start=start.replace(tzinfo=datetime.timezone.utc)
    if end.tzinfo is None:
        logger.info('Specified end is timezone-naive. Assuming UTC')
        end=end.replace(tzinfo=datetime.timezone.utc)
    if datapoint_timestamp_min is not None and datapoint_timestamp_min.tzinfo is None:
        logger.info('Specified minimum datapoint timestamp is timezone-naive. Assuming UTC')
        datapoint_timestamp_min=datapoint_timestamp_min.replace(tzinfo=datetime.timezone.utc)
    if datapoint_timestamp_max is not None and datapoint_timestamp_max.tzinfo is None:
        logger.info('Specified maximum datapoint timestamp is timezone-naive. Assuming UTC')
        datapoint_timestamp_max=datapoint_timestamp_max.replace(tzinfo=datetime.timezone.utc)
    logger.info('Downloading person position data from Honeycomb. Base directory: {}. Pose processing data subdirectory: {}. Environment ID: {}. Start: {}. End: {}'.format(
        base_dir,
        pose_processing_subdirectory,
        environment_id,
        start,
        end
    ))
    processing_start = time.time()
    logger.info('Generating metadata')
    download_position_data_metadata = generate_metadata(
        environment_id=environment_id,
        pipeline_stage='download_position_data',
        parameters={
            'datapoint_timestamp_min': datapoint_timestamp_min,
            'datapoint_timestamp_max': datapoint_timestamp_max,
            'start': start,
            'end': end
        }
    )
    download_position_data_inference_id = download_position_data_metadata.get('inference_id')
    logger.info('Writing inference metadata to local file')
    process_pose_data.local_io.write_data_local(
        data_object=download_position_data_metadata,
        base_dir=base_dir,
        pipeline_stage='download_position_data',
        environment_id=environment_id,
        filename_stem='download_position_data_metadata',
        inference_id=download_position_data_metadata['inference_id'],
        time_segment_start=None,
        object_type='dict',
        append=False,
        sort_field=None,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    logger.info('Generating list of time segments')
    time_segment_start_list = process_pose_data.local_io.generate_time_segment_start_list(
        start=start,
        end=end
    )
    num_time_segments = len(time_segment_start_list)
    num_minutes = (end - start).total_seconds()/60
    logger.info('Downloading position data for {} time segments spanning {:.3f} minutes: {} to {}'.format(
        num_time_segments,
        num_minutes,
        time_segment_start_list[0].isoformat(),
        time_segment_start_list[-1].isoformat()
    ))
    logger.info('Fetching person tag info from Honeycomb for specified environment and time span')
    person_tag_info_df = honeycomb_io.fetch_person_tag_info(
        start=start,
        end=end,
        environment_id=environment_id,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    device_ids = person_tag_info_df['device_id'].unique().tolist()
    assignment_ids = person_tag_info_df.index.tolist()
    logger.info('Found {} tags for specified environment and time span'.format(
        len(device_ids)
    ))
    if source_objects == 'position_objects':
        logger.info('Fetching position objects for these tags and specified start_end and writing to local files')
        if task_progress_bar:
            if notebook:
                time_segment_start_iterator = tqdm.notebook.tqdm(time_segment_start_list)
            else:
                time_segment_start_iterator = tqdm.tqdm(time_segment_start_list)
        else:
            time_segment_start_iterator = time_segment_start_list
        for time_segment_start in time_segment_start_iterator:
            position_data_df = honeycomb_io.fetch_cuwb_position_data(
                start=time_segment_start - datetime.timedelta(milliseconds=500),
                end=time_segment_start + datetime.timedelta(milliseconds=10500),
                device_ids=device_ids,
                environment_id=None,
                environment_name=None,
                device_types=['UWBTAG'],
                output_format='dataframe',
                sort_arguments=None,
                chunk_size=1000,
                client=client,
                uri=uri,
                token_uri=token_uri,
                audience=audience,
                client_id=client_id,
                client_secret=client_secret
            )
            # There seem to be some duplicates in honeycomb
            if position_data_df.duplicated(subset=set(position_data_df.columns).difference(['socket_read_time'])).any():
                logger.warning('Duplicate position records found in time segment {}. Deleting duplicates.'.format(
                    time_segment_start.isoformat()
                ))
                position_data_df.drop_duplicates(
                    subset=set(position_data_df.columns).difference(['socket_read_time']),
                    inplace=True
                )
            if len(position_data_df) == 0:
                continue
            position_data_df = (
                position_data_df
                .join(
                    (
                        person_tag_info_df.set_index('device_id')
                        .reindex(columns=['person_id'])
                    ),
                    on='device_id'
                )
                .rename(columns={
                    'x': 'x_position',
                    'y': 'y_position',
                    'z': 'z_position'
                })
                .reindex(columns=[
                    'timestamp',
                    'person_id',
                    'x_position',
                    'y_position',
                    'z_position'
                ])
            )
            position_data_df = process_pose_data.identify.resample_uwb_data(
                uwb_data_df=position_data_df,
                id_field_names=[
                    'person_id'
                ],
                interpolation_field_names=[
                    'x_position',
                    'y_position',
                    'z_position'
                ],
                timestamp_field_name='timestamp'
            )
            position_data_df = position_data_df.loc[
                (position_data_df['timestamp'] >= time_segment_start) &
                (position_data_df['timestamp'] < time_segment_start + datetime.timedelta(seconds=10))
            ]
            process_pose_data.local_io.write_data_local(
                data_object=position_data_df,
                base_dir=base_dir,
                pipeline_stage='download_position_data',
                environment_id=environment_id,
                filename_stem='position_data',
                inference_id=download_position_data_inference_id,
                time_segment_start=time_segment_start,
                object_type='dataframe',
                append=False,
                sort_field=None,
                pose_processing_subdirectory=pose_processing_subdirectory
            )
    elif source_objects == 'datapoints':
        logger.info('Fetching UWB datapoint IDs for these tags and specified datapoint timestamp min/max')
        data_ids = honeycomb_io.fetch_uwb_data_ids(
            datapoint_timestamp_min=datapoint_timestamp_min,
            datapoint_timestamp_max=datapoint_timestamp_max,
            assignment_ids=assignment_ids,
            chunk_size=chunk_size,
            client=None,
            uri=None,
            token_uri=None,
            audience=None,
            client_id=None,
            client_secret=None
        )
        logger.info('Found {} UWB datapoint IDs for these tags and specified datapoint timestamp min/max'.format(
            len(data_ids)
        ))
        logger.info('Fetching position data from each of these UWB datapoints and writing to local files')
        if task_progress_bar:
            if notebook:
                data_id_iterator = tqdm.notebook.tqdm(data_ids)
            else:
                data_id_iterator = tqdm.tqdm(data_ids)
        else:
            data_id_iterator = data_ids
        for data_id in data_id_iterator:
            position_data_df = honeycomb_io.fetch_uwb_data_data_id(
                data_id=data_id,
                client=None,
                uri=None,
                token_uri=None,
                audience=None,
                client_id=None,
                client_secret=None
            )
            if len(position_data_df) == 0:
                continue
            position_data_df = honeycomb_io.extract_position_data(
                df=position_data_df
            )
            if len(position_data_df) == 0:
                continue
            position_data_df = process_pose_data.identify.resample_uwb_data(
                uwb_data_df=position_data_df,
                id_field_names=[
                    'assignment_id',
                    'object_id',
                    'serial_number',
                ],
                interpolation_field_names=[
                    'x_position',
                    'y_position',
                    'z_position'
                ],
                timestamp_field_name='timestamp'
            )
            position_data_df = honeycomb_io.add_person_tag_info(
                uwb_data_df=position_data_df,
                person_tag_info_df=person_tag_info_df
            )
            for time_segment_start in time_segment_start_list:
                position_data_time_segment_df = position_data_df.loc[
                    (position_data_df['timestamp'] >= time_segment_start) &
                    (position_data_df['timestamp'] < time_segment_start + datetime.timedelta(seconds=10))
                ].reset_index(drop=True)
                if len(position_data_time_segment_df) == 0:
                    continue
                process_pose_data.local_io.write_data_local(
                    data_object=position_data_time_segment_df,
                    base_dir=base_dir,
                    pipeline_stage='download_position_data',
                    environment_id=environment_id,
                    filename_stem='position_data',
                    inference_id=download_position_data_inference_id,
                    time_segment_start=time_segment_start,
                    object_type='dataframe',
                    append=True,
                    sort_field=None,
                    pose_processing_subdirectory=pose_processing_subdirectory
                )
    else:
        raise ValueError('Source object specification \'{}\' not recognized'.format(
            source_objects
        ))
    processing_time = time.time() - processing_start
    logger.info('Downloaded {:.3f} minutes of position data in {:.3f} minutes (ratio of {:.3f})'.format(
        num_minutes,
        processing_time/60,
        (processing_time/60)/num_minutes
    ))
    return download_position_data_inference_id

def identify_pose_tracks_3d_local_by_segment(
    base_dir,
    environment_id,
    download_position_data_inference_id,
    pose_track_3d_interpolation_inference_id,
    sensor_position_keypoint_index=10,
    active_person_ids=None,
    ignore_z=False,
    min_fraction_matched=0.5,
    return_match_statistics=False,
    pose_processing_subdirectory='pose_processing',
    task_progress_bar=False,
    notebook=False
):
    """
    Fetches 3D pose and pose track data and UWB position data from local files, matches pose tracks to people, and writes output back to local files.

    Input data is assumed to be organized as specified by output of
    reconstruct_poses_3d_local_by_time_segment(),
    generate_pose_tracks_3d_local_by_time_segment(),
    interpolate_pose_tracks_3d_local_by_pose_track(), and
    download_position_data_by_datapoint().

    The script looks up the inference IDs for the 3D pose tracks and 3D poses by
    inspecting the metadata from the pose track interpolation run.

    If active person IDs are not specified, script assumes all sensors are
    assigned to people are available to match.

    Output data is saved as
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/pose_track_3d_identification/ENVIRONMENT_ID/pose_track_3d_identification_INFERENCE_ID.pkl\'.

    Output metadata is saved as
    \'BASE_DIR/POSE_PROCESSING_SUBDIRECTORY/pose_track_3d_identificationn/ENVIRONMENT_ID/pose_track_3d_identification_metadata_INFERENCE_ID.pkl\'

    Args:
        base_dir: Base directory for local data (e.g., \'/data\')
        environment_id (str): Honeycomb environment ID for source environment
        download_position_data_inference_id (str): Inference ID for source position data
        pose_track_3d_interpolation_inference_id_id (str): Inference ID for source pose track data
        sensor_position_keypoint_index (int): Index of keypoint corresponding to UWB sensor on each person (default: 10)
        active_person_ids (sequence of str): List of Honeycomb person IDs for people known to be wearing active tags (default is None)
        ignore_z (bool): Boolean indicating whether to ignore z dimension when comparing pose and sensor positions (default is False)
        min_fraction_matched (float): Minimum fraction of poses in track which must match person for track to be identified as person (default is 0.5)
        return_match_statistics (bool): Boolean indicating whether algorithm should return detailed match statistics along with inference ID (defaul is False)
        pose_processing_subdirectory (str): subdirectory (under base directory) for all pose processing data (default is \'pose_processing\')
        task_progress_bar (bool): Boolean indicating whether script should display an overall progress bar (default is False)
        notebook (bool): Boolean indicating whether script is being run in a Jupyter notebook (for progress bar display) (default is False)

    Returns:
        (str) Locally-generated inference ID for this run (identifies output data)
        (dataframe) Detailed match statistics (if requested)
    """
    pose_track_3d_interpolation_metadata = process_pose_data.local_io.fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_track_3d_interpolation',
        environment_id=environment_id,
        filename_stem='pose_track_3d_interpolation_metadata',
        inference_ids=pose_track_3d_interpolation_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=None,
        object_type='dict',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    start = pose_track_3d_interpolation_metadata['parameters']['start']
    end = pose_track_3d_interpolation_metadata['parameters']['end']
    pose_reconstruction_3d_inference_id = pose_track_3d_interpolation_metadata['parameters']['pose_reconstruction_3d_inference_id']
    pose_tracking_3d_inference_id = pose_track_3d_interpolation_metadata['parameters']['pose_tracking_3d_inference_id']
    logger.info('Identifying 3D pose tracks from local interpolated 3D pose track data and local UWB position data. Base directory: {}. Pose processing data subdirectory: {}. Environment ID: {}.'.format(
        base_dir,
        pose_processing_subdirectory,
        environment_id
    ))
    processing_start = time.time()
    logger.info('Generating metadata')
    pose_track_3d_identification_metadata = generate_metadata(
        environment_id=environment_id,
        pipeline_stage='pose_track_3d_identification',
        parameters={
            'pose_reconstruction_3d_inference_id': pose_reconstruction_3d_inference_id,
            'pose_tracking_3d_inference_id': pose_tracking_3d_inference_id,
            'pose_track_3d_interpolation_inference_id': pose_track_3d_interpolation_inference_id,
            'start': start,
            'end': end,
            'sensor_position_keypoint_index': sensor_position_keypoint_index,
            'active_person_ids': active_person_ids,
            'ignore_z': ignore_z,
            'min_fraction_matched':  min_fraction_matched,
            'return_match_statistics': return_match_statistics
        }
    )
    logger.info('Writing inference metadata to local file')
    process_pose_data.local_io.write_data_local(
        data_object=pose_track_3d_identification_metadata,
        base_dir=base_dir,
        pipeline_stage='pose_track_3d_identification',
        environment_id=environment_id,
        filename_stem='pose_track_3d_identification_metadata',
        inference_id=pose_track_3d_identification_metadata['inference_id'],
        time_segment_start=None,
        object_type='dict',
        append=False,
        sort_field=None,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    pose_track_3d_identification_inference_id = pose_track_3d_identification_metadata['inference_id']
    # Fetch pose track data
    pose_tracks_3d_before_interpolation = process_pose_data.local_io.fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_tracking_3d',
        environment_id=environment_id,
        filename_stem='pose_tracks_3d',
        inference_ids=pose_tracking_3d_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=None,
        object_type='dict',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    pose_tracks_3d_from_interpolation = process_pose_data.local_io.fetch_data_local(
        base_dir=base_dir,
        pipeline_stage='pose_tracking_3d',
        environment_id=environment_id,
        filename_stem='pose_tracks_3d',
        inference_ids=pose_track_3d_interpolation_inference_id,
        data_ids=None,
        sort_field=None,
        time_segment_start=None,
        object_type='dict',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    pose_3d_ids_with_tracks_before_interpolation_df = process_pose_data.local_io.convert_pose_tracks_3d_to_df(
        pose_tracks_3d=pose_tracks_3d_before_interpolation
    )
    pose_3d_ids_with_tracks_from_interpolation_df = process_pose_data.local_io.convert_pose_tracks_3d_to_df(
        pose_tracks_3d=pose_tracks_3d_from_interpolation
    )
    pose_3d_ids_with_tracks_df = pd.concat(
        (pose_3d_ids_with_tracks_before_interpolation_df, pose_3d_ids_with_tracks_from_interpolation_df)
    ).sort_values('pose_track_3d_id')
    logger.info('Generating list of time segments')
    time_segment_start_list = process_pose_data.local_io.generate_time_segment_start_list(
        start=start,
        end=end
    )
    num_time_segments = len(time_segment_start_list)
    num_minutes = (end - start).total_seconds()/60
    logger.info('Identifying pose tracks for {} time segments spanning {:.3f} minutes: {} to {}'.format(
        num_time_segments,
        num_minutes,
        time_segment_start_list[0].isoformat(),
        time_segment_start_list[-1].isoformat()
    ))
    if task_progress_bar:
        if notebook:
            time_segment_start_iterator = tqdm.notebook.tqdm(time_segment_start_list)
        else:
            time_segment_start_iterator = tqdm.tqdm(time_segment_start_list)
    else:
        time_segment_start_iterator = time_segment_start_list
    pose_identification_time_segment_df_list = list()
    if return_match_statistics:
        match_statistics_time_segment_df_list = list()
    for time_segment_start in time_segment_start_iterator:
        # Fetch 3D poses with tracks
        poses_3d_time_segment_df = process_pose_data.local_io.fetch_data_local(
            base_dir=base_dir,
            pipeline_stage='pose_reconstruction_3d',
            environment_id=environment_id,
            filename_stem='poses_3d',
            inference_ids=[
                pose_reconstruction_3d_inference_id,
                pose_track_3d_interpolation_inference_id
            ],
            data_ids=None,
            sort_field=None,
            time_segment_start=time_segment_start,
            object_type='dataframe',
            pose_processing_subdirectory='pose_processing'
        )
        poses_3d_with_tracks_time_segment_df = poses_3d_time_segment_df.join(pose_3d_ids_with_tracks_df, how='inner')
        # Add sensor positions
        poses_3d_with_tracks_and_sensor_positions_time_segment_df = process_pose_data.identify.extract_sensor_position_data(
            poses_3d_with_tracks_df=poses_3d_with_tracks_time_segment_df,
            sensor_position_keypoint_index=sensor_position_keypoint_index
        )
        # Fetch resampled UWB data
        uwb_data_resampled_time_segment_df = process_pose_data.local_io.fetch_data_local(
            base_dir=base_dir,
            pipeline_stage='download_position_data',
            environment_id=environment_id,
            filename_stem='position_data',
            inference_ids=download_position_data_inference_id,
            data_ids=None,
            sort_field=None,
            time_segment_start=time_segment_start,
            object_type='dataframe',
            pose_processing_subdirectory=pose_processing_subdirectory
        )
        # Identify poses
        if return_match_statistics:
            pose_identification_time_segment_df, match_statistics_time_segment_df = process_pose_data.identify.identify_poses(
                poses_3d_with_tracks_and_sensor_positions_df=poses_3d_with_tracks_and_sensor_positions_time_segment_df,
                uwb_data_resampled_df=uwb_data_resampled_time_segment_df,
                active_person_ids=active_person_ids,
                ignore_z=ignore_z,
                return_match_statistics=return_match_statistics
            )
            match_statistics_time_segment_df_list.append(match_statistics_time_segment_df)
        else:
            pose_identification_time_segment_df = process_pose_data.identify.identify_poses(
                poses_3d_with_tracks_and_sensor_positions_df=poses_3d_with_tracks_and_sensor_positions_time_segment_df,
                uwb_data_resampled_df=uwb_data_resampled_time_segment_df,
                active_person_ids=active_person_ids,
                ignore_z=ignore_z,
                return_match_statistics=return_match_statistics
            )
        # Add to list
        pose_identification_time_segment_df_list.append(pose_identification_time_segment_df)
    pose_identification_df = pd.concat(pose_identification_time_segment_df_list)
    pose_track_identification_df = process_pose_data.identify.identify_pose_tracks(
        pose_identification_df=pose_identification_df
    )
    num_poses_df = pose_3d_ids_with_tracks_df.groupby('pose_track_3d_id').size().to_frame(name='num_poses')
    pose_track_identification_df = pose_track_identification_df.join(num_poses_df, on='pose_track_3d_id')
    pose_track_identification_df['fraction_matched'] = pose_track_identification_df['max_matches']/pose_track_identification_df['num_poses']
    if min_fraction_matched is not None:
        pose_track_identification_df = pose_track_identification_df.loc[pose_track_identification_df['fraction_matched'] >= min_fraction_matched]
    process_pose_data.local_io.write_data_local(
        data_object=pose_track_identification_df,
        base_dir=base_dir,
        pipeline_stage='pose_track_3d_identification',
        environment_id=environment_id,
        filename_stem='pose_track_3d_identification',
        inference_id=pose_track_3d_identification_inference_id,
        time_segment_start=None,
        object_type='dataframe',
        append=False,
        sort_field=None,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    processing_time = time.time() - processing_start
    logger.info('Identified 3D pose tracks spanning {:.3f} {:.3f} minutes (ratio of {:.3f})'.format(
        num_minutes,
        processing_time/60,
        (processing_time/60)/num_minutes
    ))
    if return_match_statistics:
        match_statistics_df = pd.concat(match_statistics_time_segment_df_list)
        return pose_track_3d_identification_inference_id, match_statistics_df
    return pose_track_3d_identification_inference_id

def overlay_poses_2d_local(
    start,
    end,
    pose_extraction_2d_inference_id,
    base_dir,
    environment_id,
    pose_model_id,
    output_directory='./video_overlays',
    output_filename_prefix='poses_2d',
    camera_assignment_ids=None,
    camera_device_types=None,
    camera_device_ids=None,
    camera_part_numbers=None,
    camera_names=None,
    camera_serial_numbers=None,
    pose_processing_subdirectory='pose_processing',
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    local_video_directory='./videos',
    video_filename_extension='mp4',
    camera_calibrations=None,
    keypoint_connectors=None,
    pose_color='green',
    keypoint_radius=3,
    keypoint_alpha=0.6,
    keypoint_connector_alpha=0.6,
    keypoint_connector_linewidth=3,
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
    """
    Fetches 2D pose data from local files and overlays onto classroom videos.

    Fetches and overlays onto all video clips that overlap with specified start
    and end (e.g., if start is 10:32:56 and end is 10:33:20, returns videos
    starting at 10:32:50, 10:33:00 and 10:33:10).

    Script performs a logical AND across all camera specifications. If no camera
    specifications are given, returns all active cameras in environment (as
    determined by camera_device_types).

    If keypoint connectors are not specified, script uses default keypoint
    connectors for specified pose model.

    Colors can be specifed as any string interpretable by
    matplotlib.colors.to_hex().

    Input data is assumed to be organized as specified by
    extract_poses_2d_alphapose_local_by_time_segment().

    Args:
        start (datetime): Start of video overlay
        end (datetime): End of video overlay
        pose_extraction_2d_inference_id (str): Inference ID for source position data
        base_dir: Base directory for local data (e.g., \'/data\')
        environment_id (str): Honeycomb environment ID for source environment
        pose_model_id (str): Honeycomb pose model ID for pose model that defines 2D/3D pose data structure
        output_directory (str): Path to output directory (default is \'./video_overlays\')
        output_filename_prefix (str): Filename prefix for output files (default is \'poses_2d\')
        camera_assignment_ids (sequence of str): List of Honeycomb assignment IDs for target cameras (default is None)
        camera_device_types (sequence of str): List of Honeycomb device types for target cameras (default is video_io.DEFAULT_CAMERA_DEVICE_TYPES)
        camera_device_ids (sequence of str): List og Honeycomb device IDs for target cameras (default is None)
        camera_part_numbers (sequence of str): List of Honeycomb part numbers for target cameras (default is None)
        camera_names (sequence of str): List of Honeycomb device names for target cameras (default is None)
        camera_serial_numbers (sequence of str): List of Honeycomb device serial numbers for target cameras (default is None)
        pose_processing_subdirectory (str): subdirectory (under base directory) for all pose processing data (default is \'pose_processing\')
        local_video_directory (str): Path to directory where local copies of Honeycomb videos are stored (default is \'./videos\')
        video_filename_extension (str): Filename extension for local copies of Honeycomb videos (default is \'mp4\')
        camera_calibrations (dict): Dict in format {DEVICE_ID: CAMERA_CALIBRATION_DATA} (default is None)
        keypoint_connectors (array): Array of keypoints to connect with lines to form pose image (default is None)
        pose_color (str): Color of pose (default is \'green\')
        keypoint_radius (int): Radius of keypoins in pixels (default is 3)
        keypoint_alpha (float): Alpha value for keypoints (default is 0.6)
        keypoint_connector_alpha (float): Alpha value for keypoint connectors (default is 0.6)
        keypoint_connector_linewidth (float): Line width for keypoint connectors (default is 3.0)
        output_filename_datetime_format (str): Datetime format for output filename (default is \'%Y%m%d_%H%M%S_%f\')
        output_filename_extension (str): Filename extension for output (determines file format) (default is \'avi\')
        output_fourcc_string (str): FOURCC code for output format (default is \'XVID\')
        concatenate_videos (bool): Boolean indicating whether to concatenate videos for each camera into single videos (default is True)
        delete_individual_clips (bool): Boolean indicating whether to delete individual clips after concatenating (default is True)
        parallel (bool): Boolean indicating whether to use multiple parallel processes (one for each time segment) (default is False)
        num_parallel_processes (int): Number of parallel processes in pool (otherwise defaults to number of cores - 1) (default is None)
        task_progress_bar (bool): Boolean indicating whether script should display an overall progress bar (default is False)
        segment_progress_bar (bool): Boolean indicating whether script should display a progress bar for each clip (default is False)
        notebook (bool): Boolean indicating whether script is being run in a Jupyter notebook (for progress bar display) (default is False)
    """
    poses_2d_df = process_pose_data.local_io.fetch_data_local_by_time_segment(
        start=start,
        end=end,
        base_dir=base_dir,
        pipeline_stage='pose_extraction_2d',
        environment_id=environment_id,
        filename_stem='poses_2d',
        inference_ids=pose_extraction_2d_inference_id,
        data_ids=None,
        sort_field=None,
        object_type='dataframe',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    poses_2d_df = process_pose_data.local_io.convert_assignment_ids_to_camera_device_ids(poses_2d_df)
    process_pose_data.overlay.overlay_poses(
        poses_df=poses_2d_df,
        start=start,
        end=end,
        camera_assignment_ids=camera_assignment_ids,
        environment_id=environment_id,
        environment_name=None,
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
        video_filename_extension=video_filename_extension,
        pose_model_id=pose_model_id,
        camera_calibrations=None,
        pose_label_column=None,
        keypoint_connectors=keypoint_connectors,
        pose_color=pose_color,
        keypoint_radius=keypoint_radius,
        keypoint_alpha=keypoint_alpha,
        keypoint_connector_alpha=keypoint_connector_alpha,
        keypoint_connector_linewidth=keypoint_connector_linewidth,
        output_directory=output_directory,
        output_filename_prefix=output_filename_prefix,
        output_filename_datetime_format=output_filename_datetime_format,
        output_filename_extension=output_filename_extension,
        output_fourcc_string=output_fourcc_string,
        concatenate_videos=concatenate_videos,
        delete_individual_clips=delete_individual_clips,
        parallel=parallel,
        num_parallel_processes=num_parallel_processes,
        task_progress_bar=task_progress_bar,
        segment_progress_bar=segment_progress_bar,
        notebook=notebook
    )

def overlay_poses_3d_local(
    start,
    end,
    pose_reconstruction_3d_inference_id,
    base_dir,
    environment_id,
    pose_model_id,
    output_directory='./video_overlays',
    output_filename_prefix='poses_3d',
    camera_assignment_ids=None,
    camera_device_types=None,
    camera_device_ids=None,
    camera_part_numbers=None,
    camera_names=None,
    camera_serial_numbers=None,
    pose_processing_subdirectory='pose_processing',
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    local_video_directory='./videos',
    video_filename_extension='mp4',
    camera_calibrations=None,
    keypoint_connectors=None,
    pose_color='green',
    keypoint_radius=3,
    keypoint_alpha=0.6,
    keypoint_connector_alpha=0.6,
    keypoint_connector_linewidth=3,
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
    """
    Fetches 3D pose data from local files and overlays onto classroom videos.

    Fetches and overlays onto all video clips that overlap with specified start
    and end (e.g., if start is 10:32:56 and end is 10:33:20, returns videos
    starting at 10:32:50, 10:33:00 and 10:33:10).

    Script performs a logical AND across all camera specifications. If no camera
    specifications are given, returns all active cameras in environment (as
    determined by camera_device_types).

    If keypoint connectors are not specified, script uses default keypoint
    connectors for specified pose model.

    Colors can be specifed as any string interpretable by
    matplotlib.colors.to_hex().

    Input data is assumed to be organized as specified by output of
    reconstruct_poses_3d_local_by_time_segment().

    Args:
        start (datetime): Start of video overlay
        end (datetime): End of video overlay
        pose_extraction_2d_inference_id (str): Inference ID for source position data
        base_dir: Base directory for local data (e.g., \'/data\')
        environment_id (str): Honeycomb environment ID for source environment
        pose_model_id (str): Honeycomb pose model ID for pose model that defines 2D/3D pose data structure
        output_directory (str): Path to output directory (default is \'./video_overlays\')
        output_filename_prefix (str): Filename prefix for output files (default is \'poses_3d\')
        camera_assignment_ids (sequence of str): List of Honeycomb assignment IDs for target cameras (default is None)
        camera_device_types (sequence of str): List of Honeycomb device types for target cameras (default is video_io.DEFAULT_CAMERA_DEVICE_TYPES)
        camera_device_ids (sequence of str): List og Honeycomb device IDs for target cameras (default is None)
        camera_part_numbers (sequence of str): List of Honeycomb part numbers for target cameras (default is None)
        camera_names (sequence of str): List of Honeycomb device names for target cameras (default is None)
        camera_serial_numbers (sequence of str): List of Honeycomb device serial numbers for target cameras (default is None)
        pose_processing_subdirectory (str): subdirectory (under base directory) for all pose processing data (default is \'pose_processing\')
        local_video_directory (str): Path to directory where local copies of Honeycomb videos are stored (default is \'./videos\')
        video_filename_extension (str): Filename extension for local copies of Honeycomb videos (default is \'mp4\')
        camera_calibrations (dict): Dict in format {DEVICE_ID: CAMERA_CALIBRATION_DATA} (default is None)
        keypoint_connectors (array): Array of keypoints to connect with lines to form pose image (default is None)
        pose_color (str): Color of pose (default is \'green\')
        keypoint_radius (int): Radius of keypoins in pixels (default is 3)
        keypoint_alpha (float): Alpha value for keypoints (default is 0.6)
        keypoint_connector_alpha (float): Alpha value for keypoint connectors (default is 0.6)
        keypoint_connector_linewidth (float): Line width for keypoint connectors (default is 3.0)
        output_filename_datetime_format (str): Datetime format for output filename (default is \'%Y%m%d_%H%M%S_%f\')
        output_filename_extension (str): Filename extension for output (determines file format) (default is \'avi\')
        output_fourcc_string (str): FOURCC code for output format (default is \'XVID\')
        concatenate_videos (bool): Boolean indicating whether to concatenate videos for each camera into single videos (default is True)
        delete_individual_clips (bool): Boolean indicating whether to delete individual clips after concatenating (default is True)
        parallel (bool): Boolean indicating whether to use multiple parallel processes (one for each time segment) (default is False)
        num_parallel_processes (int): Number of parallel processes in pool (otherwise defaults to number of cores - 1) (default is None)
        task_progress_bar (bool): Boolean indicating whether script should display an overall progress bar (default is False)
        segment_progress_bar (bool): Boolean indicating whether script should display a progress bar for each clip (default is False)
        notebook (bool): Boolean indicating whether script is being run in a Jupyter notebook (for progress bar display) (default is False)
    """
    poses_3d_df = process_pose_data.local_io.fetch_data_local_by_time_segment(
        start=start,
        end=end,
        base_dir=base_dir,
        pipeline_stage='pose_reconstruction_3d',
        environment_id=environment_id,
        filename_stem='poses_3d',
        inference_ids=pose_reconstruction_3d_inference_id,
        data_ids=None,
        sort_field=None,
        object_type='dataframe',
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    process_pose_data.overlay.overlay_poses(
        poses_df=poses_3d_df,
        start=start,
        end=end,
        camera_assignment_ids=camera_assignment_ids,
        environment_id=environment_id,
        environment_name=None,
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
        video_filename_extension=video_filename_extension,
        pose_model_id=pose_model_id,
        camera_calibrations=None,
        pose_label_column=None,
        keypoint_connectors=keypoint_connectors,
        pose_color=pose_color,
        keypoint_radius=keypoint_radius,
        keypoint_alpha=keypoint_alpha,
        keypoint_connector_alpha=keypoint_connector_alpha,
        keypoint_connector_linewidth=keypoint_connector_linewidth,
        output_directory=output_directory,
        output_filename_prefix=output_filename_prefix,
        output_filename_datetime_format=output_filename_datetime_format,
        output_filename_extension=output_filename_extension,
        output_fourcc_string=output_fourcc_string,
        concatenate_videos=concatenate_videos,
        delete_individual_clips=delete_individual_clips,
        parallel=parallel,
        num_parallel_processes=num_parallel_processes,
        task_progress_bar=task_progress_bar,
        segment_progress_bar=segment_progress_bar,
        notebook=notebook
    )

def overlay_pose_tracks_3d_uninterpolated_local(
    start,
    end,
    pose_tracking_3d_inference_id,
    base_dir,
    environment_id,
    pose_model_id,
    output_directory='./video_overlays',
    output_filename_prefix='pose_tracks_3d_uninterpolated',
    camera_assignment_ids=None,
    camera_device_types=None,
    camera_device_ids=None,
    camera_part_numbers=None,
    camera_names=None,
    camera_serial_numbers=None,
    pose_processing_subdirectory='pose_processing',
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    local_video_directory='./videos',
    video_filename_extension='mp4',
    camera_calibrations=None,
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
    """
    Fetches uninterpolated 3D pose track data from local files and overlays onto classroom videos.

    Fetches and overlays onto all video clips that overlap with specified start
    and end (e.g., if start is 10:32:56 and end is 10:33:20, returns videos
    starting at 10:32:50, 10:33:00 and 10:33:10).

    Script performs a logical AND across all camera specifications. If no camera
    specifications are given, returns all active cameras in environment (as
    determined by camera_device_types).

    If keypoint connectors are not specified, script uses default keypoint
    connectors for specified pose model.

    Colors can be specifed as any string interpretable by
    matplotlib.colors.to_hex().

    Input data is assumed to be organized as specified by output of
    generate_pose_tracks_3d_local_by_time_segment().

    Args:
        start (datetime): Start of video overlay
        end (datetime): End of video overlay
        pose_tracking_3d_inference_id (str): Inference ID for source position data
        base_dir: Base directory for local data (e.g., \'/data\')
        environment_id (str): Honeycomb environment ID for source environment
        pose_model_id (str): Honeycomb pose model ID for pose model that defines 2D/3D pose data structure
        output_directory (str): Path to output directory (default is \'./video_overlays\')
        output_filename_prefix (str): Filename prefix for output files (default is \'pose_tracks_3d_uninterpolated\')
        camera_assignment_ids (sequence of str): List of Honeycomb assignment IDs for target cameras (default is None)
        camera_device_types (sequence of str): List of Honeycomb device types for target cameras (default is video_io.DEFAULT_CAMERA_DEVICE_TYPES)
        camera_device_ids (sequence of str): List og Honeycomb device IDs for target cameras (default is None)
        camera_part_numbers (sequence of str): List of Honeycomb part numbers for target cameras (default is None)
        camera_names (sequence of str): List of Honeycomb device names for target cameras (default is None)
        camera_serial_numbers (sequence of str): List of Honeycomb device serial numbers for target cameras (default is None)
        pose_processing_subdirectory (str): subdirectory (under base directory) for all pose processing data (default is \'pose_processing\')
        local_video_directory (str): Path to directory where local copies of Honeycomb videos are stored (default is \'./videos\')
        video_filename_extension (str): Filename extension for local copies of Honeycomb videos (default is \'mp4\')
        camera_calibrations (dict): Dict in format {DEVICE_ID: CAMERA_CALIBRATION_DATA} (default is None)
        keypoint_connectors (array): Array of keypoints to connect with lines to form pose image (default is None)
        pose_color (str): Color of pose (default is \'green\')
        keypoint_radius (int): Radius of keypoins in pixels (default is 3)
        keypoint_alpha (float): Alpha value for keypoints (default is 0.6)
        keypoint_connector_alpha (float): Alpha value for keypoint connectors (default is 0.6)
        keypoint_connector_linewidth (float): Line width for keypoint connectors (default is 3.0)
        pose_label_color (str): Color for pose label text (default is 'white')
        pose_label_background_alpha (float): Alpha value for pose label background (default is 0.6)
        pose_label_font_scale (float): Font scale for pose label (default is 1.5)
        pose_label_line_width (float): Line width for pose label text (default is 1.0)
        output_filename_datetime_format (str): Datetime format for output filename (default is \'%Y%m%d_%H%M%S_%f\')
        output_filename_extension (str): Filename extension for output (determines file format) (default is \'avi\')
        output_fourcc_string (str): FOURCC code for output format (default is \'XVID\')
        concatenate_videos (bool): Boolean indicating whether to concatenate videos for each camera into single videos (default is True)
        delete_individual_clips (bool): Boolean indicating whether to delete individual clips after concatenating (default is True)
        parallel (bool): Boolean indicating whether to use multiple parallel processes (one for each time segment) (default is False)
        num_parallel_processes (int): Number of parallel processes in pool (otherwise defaults to number of cores - 1) (default is None)
        task_progress_bar (bool): Boolean indicating whether script should display an overall progress bar (default is False)
        segment_progress_bar (bool): Boolean indicating whether script should display a progress bar for each clip (default is False)
        notebook (bool): Boolean indicating whether script is being run in a Jupyter notebook (for progress bar display) (default is False)
    """
    pose_tracks_3d_uninterpolated_df = process_pose_data.local_io.fetch_3d_poses_with_uninterpolated_tracks_local(
        base_dir=base_dir,
        environment_id=environment_id,
        pose_tracking_3d_inference_id=pose_tracking_3d_inference_id,
        start=start,
        end=end,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    pose_tracks_3d_uninterpolated_df = process_pose_data.local_io.add_short_track_labels(
        pose_tracks_3d_uninterpolated_df,
        pose_track_3d_id_column_name='pose_track_3d_id'
    )
    process_pose_data.overlay.overlay_poses(
        poses_df=pose_tracks_3d_uninterpolated_df,
        start=start,
        end=end,
        camera_assignment_ids=camera_assignment_ids,
        environment_id=environment_id,
        environment_name=None,
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
        video_filename_extension=video_filename_extension,
        pose_model_id=pose_model_id,
        camera_calibrations=None,
        pose_label_column='pose_track_3d_id_short',
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
        concatenate_videos=concatenate_videos,
        delete_individual_clips=delete_individual_clips,
        parallel=parallel,
        num_parallel_processes=num_parallel_processes,
        task_progress_bar=task_progress_bar,
        segment_progress_bar=segment_progress_bar,
        notebook=notebook
    )

def overlay_pose_tracks_3d_interpolated_local(
    start,
    end,
    pose_track_3d_interpolation_inference_id,
    base_dir,
    environment_id,
    pose_model_id,
    output_directory='./video_overlays',
    output_filename_prefix='pose_tracks_3d_interpolated',
    camera_assignment_ids=None,
    camera_device_types=None,
    camera_device_ids=None,
    camera_part_numbers=None,
    camera_names=None,
    camera_serial_numbers=None,
    pose_processing_subdirectory='pose_processing',
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    local_video_directory='./videos',
    video_filename_extension='mp4',
    camera_calibrations=None,
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
    """
    Fetches interpolated 3D pose track data from local files and overlays onto classroom videos.

    Fetches and overlays onto all video clips that overlap with specified start
    and end (e.g., if start is 10:32:56 and end is 10:33:20, returns videos
    starting at 10:32:50, 10:33:00 and 10:33:10).

    Script performs a logical AND across all camera specifications. If no camera
    specifications are given, returns all active cameras in environment (as
    determined by camera_device_types).

    If keypoint connectors are not specified, script uses default keypoint
    connectors for specified pose model.

    Colors can be specifed as any string interpretable by
    matplotlib.colors.to_hex().

    Input data is assumed to be organized as specified by output of
    interpolate_pose_tracks_3d_local_by_pose_track().

    Args:
        start (datetime): Start of video overlay
        end (datetime): End of video overlay
        pose_track_3d_interpolation_inference_id (str): Inference ID for source position data
        base_dir: Base directory for local data (e.g., \'/data\')
        environment_id (str): Honeycomb environment ID for source environment
        pose_model_id (str): Honeycomb pose model ID for pose model that defines 2D/3D pose data structure
        output_directory (str): Path to output directory (default is \'./video_overlays\')
        output_filename_prefix (str): Filename prefix for output files (default is \'pose_tracks_3d_interpolated\')
        camera_assignment_ids (sequence of str): List of Honeycomb assignment IDs for target cameras (default is None)
        camera_device_types (sequence of str): List of Honeycomb device types for target cameras (default is video_io.DEFAULT_CAMERA_DEVICE_TYPES)
        camera_device_ids (sequence of str): List og Honeycomb device IDs for target cameras (default is None)
        camera_part_numbers (sequence of str): List of Honeycomb part numbers for target cameras (default is None)
        camera_names (sequence of str): List of Honeycomb device names for target cameras (default is None)
        camera_serial_numbers (sequence of str): List of Honeycomb device serial numbers for target cameras (default is None)
        pose_processing_subdirectory (str): subdirectory (under base directory) for all pose processing data (default is \'pose_processing\')
        local_video_directory (str): Path to directory where local copies of Honeycomb videos are stored (default is \'./videos\')
        video_filename_extension (str): Filename extension for local copies of Honeycomb videos (default is \'mp4\')
        camera_calibrations (dict): Dict in format {DEVICE_ID: CAMERA_CALIBRATION_DATA} (default is None)
        keypoint_connectors (array): Array of keypoints to connect with lines to form pose image (default is None)
        pose_color (str): Color of pose (default is \'green\')
        keypoint_radius (int): Radius of keypoins in pixels (default is 3)
        keypoint_alpha (float): Alpha value for keypoints (default is 0.6)
        keypoint_connector_alpha (float): Alpha value for keypoint connectors (default is 0.6)
        keypoint_connector_linewidth (float): Line width for keypoint connectors (default is 3.0)
        pose_label_color (str): Color for pose label text (default is 'white')
        pose_label_background_alpha (float): Alpha value for pose label background (default is 0.6)
        pose_label_font_scale (float): Font scale for pose label (default is 1.5)
        pose_label_line_width (float): Line width for pose label text (default is 1.0)
        output_filename_datetime_format (str): Datetime format for output filename (default is \'%Y%m%d_%H%M%S_%f\')
        output_filename_extension (str): Filename extension for output (determines file format) (default is \'avi\')
        output_fourcc_string (str): FOURCC code for output format (default is \'XVID\')
        concatenate_videos (bool): Boolean indicating whether to concatenate videos for each camera into single videos (default is True)
        delete_individual_clips (bool): Boolean indicating whether to delete individual clips after concatenating (default is True)
        parallel (bool): Boolean indicating whether to use multiple parallel processes (one for each time segment) (default is False)
        num_parallel_processes (int): Number of parallel processes in pool (otherwise defaults to number of cores - 1) (default is None)
        task_progress_bar (bool): Boolean indicating whether script should display an overall progress bar (default is False)
        segment_progress_bar (bool): Boolean indicating whether script should display a progress bar for each clip (default is False)
        notebook (bool): Boolean indicating whether script is being run in a Jupyter notebook (for progress bar display) (default is False)
    """
    pose_tracks_3d_interpolated_df = process_pose_data.local_io.fetch_3d_poses_with_interpolated_tracks_local(
        base_dir=base_dir,
        environment_id=environment_id,
        pose_track_3d_interpolation_inference_id=pose_track_3d_interpolation_inference_id,
        start=start,
        end=end,
        pose_processing_subdirectory=pose_processing_subdirectory
    )
    pose_tracks_3d_interpolated_df = process_pose_data.local_io.add_short_track_labels(
        pose_tracks_3d_interpolated_df,
        pose_track_3d_id_column_name='pose_track_3d_id'
    )
    process_pose_data.overlay.overlay_poses(
        poses_df=pose_tracks_3d_interpolated_df ,
        start=start,
        end=end,
        camera_assignment_ids=camera_assignment_ids,
        environment_id=environment_id,
        environment_name=None,
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
        video_filename_extension=video_filename_extension,
        pose_model_id=pose_model_id,
        camera_calibrations=None,
        pose_label_column='pose_track_3d_id_short',
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
        concatenate_videos=concatenate_videos,
        delete_individual_clips=delete_individual_clips,
        parallel=parallel,
        num_parallel_processes=num_parallel_processes,
        task_progress_bar=task_progress_bar,
        segment_progress_bar=segment_progress_bar,
        notebook=notebook
    )

def overlay_pose_tracks_3d_identified_interpolated_local(
    start,
    end,
    pose_track_3d_identification_inference_id,
    base_dir,
    environment_id,
    pose_model_id,
    output_directory='./video_overlays',
    output_filename_prefix='pose_tracks_3d_identified_interpolated',
    camera_assignment_ids=None,
    camera_device_types=None,
    camera_device_ids=None,
    camera_part_numbers=None,
    camera_names=None,
    camera_serial_numbers=None,
    pose_track_label_column='short_name',
    pose_processing_subdirectory='pose_processing',
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    local_video_directory='./videos',
    video_filename_extension='mp4',
    camera_calibrations=None,
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
    """
    Fetches identified, interpolated 3D pose track data from local files and overlays onto classroom videos.

    Fetches and overlays onto all video clips that overlap with specified start
    and end (e.g., if start is 10:32:56 and end is 10:33:20, returns videos
    starting at 10:32:50, 10:33:00 and 10:33:10).

    Script performs a logical AND across all camera specifications. If no camera
    specifications are given, returns all active cameras in environment (as
    determined by camera_device_types).

    If keypoint connectors are not specified, script uses default keypoint
    connectors for specified pose model.

    Colors can be specifed as any string interpretable by
    matplotlib.colors.to_hex().

    Input data is assumed to be organized as specified by output of
    identify_pose_tracks_3d_local_by_segment().

    Args:
        start (datetime): Start of video overlay
        end (datetime): End of video overlay
        pose_track_3d_identification_inference_id (str): Inference ID for source position data
        base_dir: Base directory for local data (e.g., \'/data\')
        environment_id (str): Honeycomb environment ID for source environment
        pose_model_id (str): Honeycomb pose model ID for pose model that defines 2D/3D pose data structure
        output_directory (str): Path to output directory (default is \'./video_overlays\')
        output_filename_prefix (str): Filename prefix for output files (default is \'pose_tracks_3d_identified_interpolated\')
        camera_assignment_ids (sequence of str): List of Honeycomb assignment IDs for target cameras (default is None)
        camera_device_types (sequence of str): List of Honeycomb device types for target cameras (default is video_io.DEFAULT_CAMERA_DEVICE_TYPES)
        camera_device_ids (sequence of str): List og Honeycomb device IDs for target cameras (default is None)
        camera_part_numbers (sequence of str): List of Honeycomb part numbers for target cameras (default is None)
        camera_names (sequence of str): List of Honeycomb device names for target cameras (default is None)
        camera_serial_numbers (sequence of str): List of Honeycomb device serial numbers for target cameras (default is None)
        pose_track_label_column (str): Name of person data column to use for pose labels (default is \'short_name\')
        pose_processing_subdirectory (str): subdirectory (under base directory) for all pose processing data (default is \'pose_processing\')
        local_video_directory (str): Path to directory where local copies of Honeycomb videos are stored (default is \'./videos\')
        video_filename_extension (str): Filename extension for local copies of Honeycomb videos (default is \'mp4\')
        camera_calibrations (dict): Dict in format {DEVICE_ID: CAMERA_CALIBRATION_DATA} (default is None)
        keypoint_connectors (array): Array of keypoints to connect with lines to form pose image (default is None)
        pose_color (str): Color of pose (default is \'green\')
        keypoint_radius (int): Radius of keypoins in pixels (default is 3)
        keypoint_alpha (float): Alpha value for keypoints (default is 0.6)
        keypoint_connector_alpha (float): Alpha value for keypoint connectors (default is 0.6)
        keypoint_connector_linewidth (float): Line width for keypoint connectors (default is 3.0)
        pose_label_color (str): Color for pose label text (default is 'white')
        pose_label_background_alpha (float): Alpha value for pose label background (default is 0.6)
        pose_label_font_scale (float): Font scale for pose label (default is 1.5)
        pose_label_line_width (float): Line width for pose label text (default is 1.0)
        output_filename_datetime_format (str): Datetime format for output filename (default is \'%Y%m%d_%H%M%S_%f\')
        output_filename_extension (str): Filename extension for output (determines file format) (default is \'avi\')
        output_fourcc_string (str): FOURCC code for output format (default is \'XVID\')
        concatenate_videos (bool): Boolean indicating whether to concatenate videos for each camera into single videos (default is True)
        delete_individual_clips (bool): Boolean indicating whether to delete individual clips after concatenating (default is True)
        parallel (bool): Boolean indicating whether to use multiple parallel processes (one for each time segment) (default is False)
        num_parallel_processes (int): Number of parallel processes in pool (otherwise defaults to number of cores - 1) (default is None)
        task_progress_bar (bool): Boolean indicating whether script should display an overall progress bar (default is False)
        segment_progress_bar (bool): Boolean indicating whether script should display a progress bar for each clip (default is False)
        notebook (bool): Boolean indicating whether script is being run in a Jupyter notebook (for progress bar display) (default is False)
    """
    pose_tracks_3d_identified_interpolated_df = process_pose_data.local_io.fetch_3d_poses_with_person_info(
        base_dir=base_dir,
        environment_id=environment_id,
        pose_track_3d_identification_inference_id=pose_track_3d_identification_inference_id,
        start=start,
        end=end,
        pose_processing_subdirectory=pose_processing_subdirectory,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    process_pose_data.overlay.overlay_poses(
        poses_df=pose_tracks_3d_identified_interpolated_df,
        start=start,
        end=end,
        camera_assignment_ids=camera_assignment_ids,
        environment_id=environment_id,
        environment_name=None,
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
        video_filename_extension=video_filename_extension,
        pose_model_id=pose_model_id,
        camera_calibrations=None,
        pose_label_column=pose_track_label_column,
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
        concatenate_videos=concatenate_videos,
        delete_individual_clips=delete_individual_clips,
        parallel=parallel,
        num_parallel_processes=num_parallel_processes,
        task_progress_bar=task_progress_bar,
        segment_progress_bar=segment_progress_bar,
        notebook=notebook
    )

def generate_metadata(
    environment_id,
    pipeline_stage,
    parameters
):
    metadata = {
        'inference_id': uuid4().hex,
        'infererence_execution_start': datetime.datetime.now(tz=datetime.timezone.utc),
        'inference_execution_name': pipeline_stage,
        'inference_execution_model': 'wf-process-pose-data',
        'inference_execution_version': process_pose_data.__version__,
        'parameters': parameters
    }
    return metadata

def extract_coordinate_space_id_from_camera_calibrations(camera_calibrations):
    coordinate_space_ids = set([camera_calibration.get('space_id') for camera_calibration in camera_calibrations.values()])
    if len(coordinate_space_ids) > 1:
        raise ValueError('Multiple coordinate space IDs found in camera calibration data')
    coordinate_space_id = list(coordinate_space_ids)[0]
    return coordinate_space_id

def generate_pose_3d_limits(
    pose_model_id,
    room_x_limits,
    room_y_limits,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    pose_model = honeycomb_io.fetch_pose_model_by_pose_model_id(
        pose_model_id,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    pose_model_name = pose_model.get('model_name')
    pose_3d_limits = process_pose_data.reconstruct.pose_3d_limits_by_pose_model(
        room_x_limits=room_x_limits,
        room_y_limits=room_y_limits,
        pose_model_name=pose_model_name
    )
    return pose_3d_limits
