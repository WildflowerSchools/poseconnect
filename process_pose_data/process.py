import process_pose_data.local_io
import process_pose_data.honeycomb_io
import process_pose_data.analyze
import tqdm
from uuid import uuid4
import multiprocessing
import functools
import logging
import datetime
import time

logger = logging.getLogger(__name__)

def reconstruct_poses_3d_alphapose_local_by_time_segment(
    start,
    end,
    base_dir,
    environment_id,
    pose_model_id,
    room_x_limits,
    room_y_limits,
    honeycomb_inference_execution=False,
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
    poses_2d_file_name='alphapose-results.json',
    poses_2d_json_format='cmu',
    poses_3d_directory_name='poses_3d',
    poses_3d_file_name_stem='poses_3d',
    inference_metadata_filename_stem='inference_metadata',
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
    progress_bar=False,
    notebook=False
):
    if start.tzinfo is None:
        logger.info('Specified start is timezone-naive. Assuming UTC')
        start=start.replace(tzinfo=datetime.timezone.utc)
    if end.tzinfo is None:
        logger.info('Specified end is timezone-naive. Assuming UTC')
        end=end.replace(tzinfo=datetime.timezone.utc)
    logger.info('Reconstructing 3D poses from local 2D pose data. Base directory: {}. Environment ID: {}. Start: {}. End: {}'.format(
        base_dir,
        environment_id,
        start,
        end
    ))
    logger.info('Generating inference metadata')
    inference_metadata = generate_inference_metadata_reconstruct_3d_poses_alphapose_local(
        start=start,
        end=end,
        environment_id=environment_id,
        pose_model_id=pose_model_id,
        pose_3d_limits=pose_3d_limits,
        room_x_limits=room_x_limits,
        room_y_limits=room_y_limits,
        honeycomb_inference_execution=honeycomb_inference_execution,
        camera_assignment_ids=camera_assignment_ids,
        camera_device_id_lookup=camera_device_id_lookup,
        camera_calibrations=camera_calibrations,
        coordinate_space_id=coordinate_space_id,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    inference_id = inference_metadata.get('inference_execution').get('inference_id')
    camera_assignment_ids = inference_metadata.get('camera_assignment_ids')
    camera_device_id_lookup = inference_metadata.get('camera_device_id_lookup')
    camera_device_ids = inference_metadata.get('camera_device_ids')
    camera_calibrations = inference_metadata.get('camera_calibrations')
    pose_3d_limits = inference_metadata.get('pose_3d_limits')
    logger.info('Writing inference metadata to local file')
    process_pose_data.local_io.write_inference_metadata_local(
        inference_metadata=inference_metadata,
        base_dir=base_dir,
        environment_id=environment_id,
        subdirectory_name=poses_3d_directory_name,
        inference_metadata_filename_stem=inference_metadata_filename_stem
    )
    logger.info('Generating list of time segments')
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
    reconstruct_poses_3d_alphapose_local_time_segment_partial = functools.partial(
        reconstruct_poses_3d_alphapose_local_time_segment,
        base_dir=base_dir,
        environment_id=environment_id,
        poses_3d_inference_id=inference_id,
        poses_2d_file_name=poses_2d_file_name,
        poses_2d_json_format=poses_2d_json_format,
        poses_3d_directory_name=poses_3d_directory_name,
        poses_3d_file_name_stem=poses_3d_file_name_stem,
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
    if progress_bar and parallel and ~notebook:
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
            poses_3d_df_list = p.map(reconstruct_poses_3d_alphapose_local_time_segment_partial, time_segment_start_list)
    else:
        poses_3d_df_list = list(map(reconstruct_poses_3d_alphapose_local_time_segment_partial, time_segment_start_list))
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
    poses_3d_inference_id,
    poses_2d_file_name='alphapose-results.json',
    poses_2d_json_format='cmu',
    poses_3d_directory_name='poses_3d',
    poses_3d_file_name_stem='poses_3d',
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
        file_name=poses_2d_file_name,
        json_format=poses_2d_json_format
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
        inference_id=poses_3d_inference_id,
        directory_name=poses_3d_directory_name,
        file_name_stem=poses_3d_file_name_stem
    )

def upload_3d_poses_honeycomb(
    inference_id,
    base_dir,
    environment_id,
    poses_3d_directory_name='poses_3d',
    poses_3d_file_name_stem='poses_3d',
    inference_metadata_filename_stem='inference_metadata',
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    progress_bar=False,
    notebook=False
):
    inference_metadata = process_pose_data.local_io.read_inference_metadata_local(
        inference_id=inference_id,
        base_dir=base_dir,
        environment_id=environment_id,
        subdirectory_name=poses_3d_directory_name,
        inference_metadata_filename_stem=inference_metadata_filename_stem
    )
    start = inference_metadata.get('start')
    end = inference_metadata.get('end')
    pose_model_id = inference_metadata.get('pose_model_id')
    coordinate_space_id = inference_metadata.get('coordinate_space_id')
    time_segment_start_list = process_pose_data.local_io.generate_time_segment_start_list(
        start=start,
        end=end
    )
    num_time_segments = len(time_segment_start_list)
    num_minutes = (end - start).seconds/60
    logger.info('Uploading 3D poses to Honeycomb for {} time segments spanning {:.3f} minutes: {} to {}'.format(
        num_time_segments,
        num_minutes,
        time_segment_start_list[0].isoformat(),
        time_segment_start_list[-1].isoformat()
    ))
    pose_ids_3d=list()
    if progress_bar:
        if notebook:
            time_segment_start_iterator = tqdm.notebook.tqdm(time_segment_start_list)
        else:
            time_segment_start_iterator = tqdm.tqdm(time_segment_start_list)
    else:
        time_segment_start_iterator = time_segment_start_list
    pose_3d_ids=list()
    for time_segment_start in time_segment_start_iterator:
        poses_3d_df_time_segment = process_pose_data.local_io.fetch_3d_pose_data_local_time_segment(
            time_segment_start=time_segment_start,
            base_dir=base_dir,
            environment_id=environment_id,
            inference_id=inference_id,
            directory_name=poses_3d_directory_name,
            file_name_stem=poses_3d_file_name_stem
        )
        pose_ids_3d_time_segment = process_pose_data.honeycomb_io.write_3d_pose_data(
            poses_3d_df=poses_3d_df_time_segment,
            coordinate_space_id=coordinate_space_id,
            pose_model_id=pose_model_id,
            source_id=inference_id,
            source_type='INFERRED',
            chunk_size=chunk_size,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
        pose_3d_ids.extend(pose_ids_3d_time_segment)
    return pose_3d_ids

def delete_reconstruct_3d_poses_output(
    base_dir,
    environment_id,
    inference_id,
    poses_3d_directory_name='poses_3d',
    poses_3d_file_name_stem='poses_3d',
    inference_metadata_filename_stem='inference_metadata',
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    logger.info('Deleting local 3D pose data')
    process_pose_data.local_io.delete_3d_pose_data_local(
        base_dir=base_dir,
        environment_id=environment_id,
        inference_id=inference_id,
        directory_name=poses_3d_directory_name,
        file_name_stem=poses_3d_file_name_stem
    )
    logger.info('Deleting local inference metadata')
    process_pose_data.local_io.delete_inference_metadata_local(
        inference_id=inference_id,
        base_dir=base_dir,
        environment_id=environment_id,
        subdirectory_name=poses_3d_directory_name,
        inference_metadata_filename_stem=inference_metadata_filename_stem
    )
    logger.info('Deleting Honeycomb 3D pose data')
    process_pose_data.honeycomb_io.delete_3d_pose_data_by_inference_id(
        inference_id=inference_id,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    logger.info('Deleting Honeycomb inference execution object')
    process_pose_data.honeycomb_io.delete_inference_execution(
        inference_id=inference_id,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )

def generate_inference_metadata_reconstruct_3d_poses_alphapose_local(
    start,
    end,
    environment_id,
    pose_model_id,
    pose_3d_limits,
    room_x_limits,
    room_y_limits,
    honeycomb_inference_execution=False,
    camera_assignment_ids=None,
    camera_device_id_lookup=None,
    camera_calibrations=None,
    coordinate_space_id=None,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    logger.info('Generating inference execution object')
    inference_execution = generate_inference_execution_reconstruct_3d_poses_alphapose_local(
        environment_id,
        start,
        end,
        honeycomb_inference_execution,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    if camera_assignment_ids is None:
        logger.info('Camera assignment IDs not specified. Fetching camera assignment IDs from Honeycomb based on environmen and time span')
        camera_assignment_ids = process_pose_data.honeycomb_io.fetch_camera_assignment_ids_from_environment(
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
        logger.info('Camera calibration parameters not specified. Fetching camera calibration parameters from Honeycomb based on camera device IDs and time span')
        camera_calibrations = process_pose_data.honeycomb_io.fetch_camera_calibrations(
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
    inference_metadata = {
        'start': start,
        'end': end,
        'environment_id': environment_id,
        'pose_model_id': pose_model_id,
        'coordinate_space_id': coordinate_space_id,
        'inference_execution': inference_execution,
        'camera_assignment_ids': camera_assignment_ids,
        'camera_device_id_lookup': camera_device_id_lookup,
        'camera_device_ids': camera_device_ids,
        'camera_calibrations': camera_calibrations,
        'pose_3d_limits': pose_3d_limits
    }
    return inference_metadata

def generate_inference_execution_reconstruct_3d_poses_alphapose_local(
    environment_id,
    start,
    end,
    honeycomb_inference_execution=False,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    inference_execution_start = datetime.datetime.now(tz=datetime.timezone.utc)
    inference_execution_name = 'Reconstruct 3D poses from 2D poses'
    inference_execution_notes = 'Environment: {} Start: {} End: {}'.format(
        environment_id,
        start.isoformat(),
        end.isoformat()
    )
    inference_execution_model = 'process_pose_data.process.reconstruct_poses_3d_alphapose_local_by_time_segment'
    inference_execution_version = '2.4.0'
    if honeycomb_inference_execution:
        logger.info('Writing inference execution info to Honeycomb')
        inference_id = process_pose_data.honeycomb_io.create_inference_execution(
            execution_start=inference_execution_start,
            name=inference_execution_name,
            notes=inference_execution_notes,
            model=inference_execution_model,
            version=inference_execution_version,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    else:
        logger.info('Generating local inference execution ID')
        inference_id = uuid4().hex
    inference_execution = {
        'inference_id': inference_id,
        'name': inference_execution_name,
        'notes': inference_execution_notes,
        'model': inference_execution_model,
        'version': inference_execution_version,
        'execution_start': inference_execution_start,

    }
    return inference_execution

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
    return pose_3d_limits
