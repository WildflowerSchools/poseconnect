import process_pose_data.core
import geom_render
import minimal_honeycomb
import pandas as pd
import numpy as np
import tqdm
import datetime
import os
import logging

logger = logging.getLogger(__name__)

def fetch_geoms_2d_by_inference_execution(
    inference_id=None,
    inference_name=None,
    inference_model=None,
    inference_version=None,
    chunk_size=100,
    frames_per_second=10.0,
    track_label_color='ff0000',
    track_label_alpha=1.0,
    keypoint_color='00ff00',
    keypoint_alpha=0.3,
    pose_line_color='00ff00',
    pose_line_alpha=0.3,
    progress_bar=False,
    notebook=False,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    logger.info('Fetching pose data')
    df = process_pose_data.core.fetch_2d_pose_data_by_inference_execution(
        inference_id=inference_id,
        inference_name=inference_name,
        inference_model=inference_model,
        inference_version=inference_version,
        chunk_size=chunk_size,
        progress_bar=progress_bar,
        notebook=notebook,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    logger.info('Creating geom collections')
    geom_collection_2d_dict = create_geom_collection_2d_dict(
        df,
        frames_per_second=frames_per_second,
        track_label_color=track_label_color,
        track_label_alpha=track_label_alpha,
        keypoint_color=keypoint_color,
        keypoint_alpha=keypoint_alpha,
        pose_line_color=pose_line_color,
        pose_line_alpha=pose_line_alpha,
        progress_bar=progress_bar,
        notebook=notebook,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    return geom_collection_2d_dict

def fetch_geoms_2d_by_time_span(
    environment_name,
    start_time,
    end_time=None,
    camera_device_types=['PI3WITHCAMERA'],
    chunk_size=100,
    frames_per_second=10.0,
    track_label_color='ff0000',
    track_label_alpha=1.0,
    keypoint_color='00ff00',
    keypoint_alpha=0.3,
    pose_line_color='00ff00',
    pose_line_alpha=0.3,
    progress_bar=False,
    notebook=False,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    logger.info('Fetching pose data')
    df = process_pose_data.core.fetch_2d_pose_data_by_time_span(
        environment_name=environment_name,
        start_time=start_time,
        end_time=end_time,
        camera_device_types=camera_device_types,
        chunk_size=chunk_size,
        progress_bar=progress_bar,
        notebook=notebook,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    logger.info('Creating geom collections')
    geom_collection_2d_dict = create_geom_collection_2d_dict(
        df,
        frames_per_second=frames_per_second,
        track_label_color=track_label_color,
        track_label_alpha=track_label_alpha,
        keypoint_color=keypoint_color,
        keypoint_alpha=keypoint_alpha,
        pose_line_color=pose_line_color,
        pose_line_alpha=pose_line_alpha,
        progress_bar=progress_bar,
        notebook=notebook,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    return geom_collection_2d_dict

def create_geom_collection_2d_dict(
    df,
    frames_per_second=10.0,
    track_label_color='ff0000',
    track_label_alpha=1.0,
    keypoint_color='00ff00',
    keypoint_alpha=0.3,
    pose_line_color='00ff00',
    pose_line_alpha=0.3,
    progress_bar=False,
    notebook=False,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    logger.info('Extracting pose model ID from data')
    pose_model_id = process_pose_data.core.extract_pose_model_id(df)
    logger.info('Fetching pose model info')
    pose_model_info = process_pose_data.core.fetch_pose_model_info(
        pose_model_id=pose_model_id,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    num_keypoints = len(pose_model_info['keypoint_names'])
    num_keypoints_values = df['keypoint_array'].apply(lambda x: x.shape[0]).unique().tolist()
    if len(num_keypoints_values) > 1:
        raise ValueError('Data contains multiple values for number of keypoints per pose: {}'.format(
            num_keypoints_values
        ))
    if num_keypoints_values[0] != num_keypoints:
        raise ValueError('Pose model specifies {} keypoints per pose but data contains {} keypoints per pose'.format(
            num_keypoints,
            num_keypoints_values[0]
        ))
    num_spatial_dimensions_values = df['keypoint_array'].apply(lambda x: x.shape[1]).unique().tolist()
    if len(num_spatial_dimensions_values) > 1:
        raise ValueError('Data contains multiple values for number of spatial dimensions per keypoint: {}'.format(
            num_spatial_dimensions_values
        ))
    if num_spatial_dimensions_values[0] != 2:
        raise ValueError('Function expects 2 spatial dimensions per keypoint but data contains {} spatial dimensions per keypoint'.format(
            num_spatial_dimensions_values[0]
        ))
    keypoint_conectors = pose_model_info['keypoint_connectors']
    num_cameras = len(df['camera_device_id'].unique())
    logger.info('Creating geom collections for each {} camera views'.format(
        num_cameras
    ))
    geom_collection_2d_dict = dict()
    for camera_device_id, group_df in df.groupby('camera_device_id'):
        geom_collection_2d_dict[camera_device_id] = create_geom_collection_2d(
            df=group_df,
            num_keypoints=num_keypoints,
            keypoint_connectors=keypoint_conectors,
            frames_per_second=frames_per_second,
            track_label_color=track_label_color,
            track_label_alpha=track_label_alpha,
            keypoint_color=keypoint_color,
            keypoint_alpha=keypoint_alpha,
            pose_line_color=pose_line_color,
            pose_line_alpha=pose_line_alpha,
            progress_bar=progress_bar,
            notebook=notebook
        )
    return geom_collection_2d_dict

def create_geom_collection_2d(
    df,
    num_keypoints,
    keypoint_connectors,
    frames_per_second=10.0,
    track_label_color='ff0000',
    track_label_alpha=1.0,
    keypoint_color='00ff00',
    keypoint_alpha=0.3,
    pose_line_color='00ff00',
    pose_line_alpha=0.3,
    progress_bar=False,
    notebook=False
):
    num_spatial_dimensions = 2
    # Extract camera device id
    camera_device_ids = df['camera_device_id'].unique().tolist()
    if len(camera_device_ids) > 1:
        raise ValueError('Data contains multiple camera device IDs: {}'.format(
            camera_device_ids
        ))
    else:
        camera_device_id = camera_device_ids[0]
    # Extract camera name
    camera_names = df['camera_name'].unique().tolist()
    if len(camera_names) > 1:
        raise ValueError('Data contains multiple camera names: {}'.format(
            camera_names
        ))
    else:
        camera_name = camera_names[0]
    # Calculate time index information
    time_between_frames = datetime.timedelta(microseconds=10**6/frames_per_second)
    min_timestamp = df['timestamp'].min()
    max_timestamp = df['timestamp'].max()
    start_time = min_timestamp.to_pydatetime()
    end_time = max_timestamp.to_pydatetime()
    num_frames = int(round((end_time - start_time)/time_between_frames)) + 1
    # Extract track labels
    track_labels = np.sort(df['track_label'].unique()).tolist()
    num_tracks = len(track_labels)
    # Define track index lookup
    track_index_lookup = {track_label: track_index for track_index, track_label in enumerate(track_labels)}
    # Extract coordinates
    logger.info('Extracting coordinates for {} tracks across {} frames for camera {}'.format(
        num_tracks,
        num_frames,
        camera_name
    ))
    coordinates = np.full((num_frames, num_tracks*(num_keypoints + 1), num_spatial_dimensions), np.nan)
    if progress_bar:
        if notebook:
            dataframe_iterable = tqdm.tqdm_notebook(df.iterrows(), total=len(df))
        else:
            dataframe_iterable = tqdm.tqdm(df.iterrows(), total=len(df))
    else:
        dataframe_iterable = df.iterrows()
    for index, row in dataframe_iterable:
        frame_index = int(round((row['timestamp'].to_pydatetime() - min_timestamp)/time_between_frames))
        track_index = track_index_lookup[row['track_label']]
        keypoint_array = row['keypoint_array']
        centroid = row['centroid']
        base_coordinate_index = track_index*(num_keypoints + 1)
        coordinates[frame_index, base_coordinate_index:(base_coordinate_index + num_keypoints)] = keypoint_array
        coordinates[frame_index, base_coordinate_index + num_keypoints] = centroid
    # Define geom list
    logger.info('Defining geom list')
    geom_list = []
    for track_index in range(num_tracks):
        base_coordinate_index = track_index*(num_keypoints + 1)
        track_label = track_labels[track_index]
        # Track label
        geom_list.append(
            geom_render.Text2D(
                coordinate_indices=[base_coordinate_index + num_keypoints],
                text=str(track_label),
                color=track_label_color,
                alpha=track_label_alpha
            )
        )
        # Points to represent keypoints
        for keypoint_index in range(num_keypoints):
            geom_list.append(
                geom_render.Point2D(
                    coordinate_indices=[base_coordinate_index + keypoint_index],
                    color=keypoint_color,
                    alpha=keypoint_alpha
                )
            )
        # Lines to represent keypoint connectors
        for keypoint_connector in keypoint_connectors:
            geom_list.append(
                geom_render.Line2D(
                    coordinate_indices = [
                        base_coordinate_index + keypoint_connector[0],
                        base_coordinate_index + keypoint_connector[1]
                    ],
                    color=pose_line_color,
                    alpha=pose_line_alpha
                )
            )
    geom_collection_2d = geom_render.GeomCollection2D(
        start_time=start_time,
        frames_per_second=frames_per_second,
        num_frames=num_frames,
        coordinates = coordinates,
        geom_list = geom_list
    )
    return geom_collection_2d
