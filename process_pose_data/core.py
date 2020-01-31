import minimal_honeycomb
import pandas as pd
import numpy as np
import datetime
import logging

logger = logging.getLogger(__name__)

def fetch_2d_pose_data_by_inference_execution(
    inference_id=None,
    inference_name=None,
    inference_model=None,
    inference_version=None,
    chunk_size=100,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    client = minimal_honeycomb.MinimalHoneycombClient(
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    if inference_id is not None:
        if inference_name is not None or inference_model is not None or inference_version is not None:
            raise ValueError('Must specify either inference ID or inference name/model/version but not both')
    else:
        if inference_name is None and inference_model is None and inference_version is None:
            raise ValueError('If inference ID is not specified, must specify at least one of inference name/model/version')
        arguments = dict()
        if inference_name is not None:
            arguments['name'] = {
                'type': 'String',
                'value': inference_name
            }
        if inference_model is not None:
            arguments['model'] = {
                'type': 'String',
                'value': inference_model
            }
        if inference_version is not None:
            arguments['version'] = {
                'type': 'String',
                'value': inference_version
            }
        logger.info('Finding inference execution runs that match the specified name/model/version')
        result = client.bulk_query(
            request_name='findInferenceExecutions',
            arguments=arguments,
            return_data=[
                'inference_id',
            ],
            id_field_name='inference_id'
        )
        if len(result) == 0:
            raise ValueError('No inference executions match specified name/model/version')
        if len(result) > 1:
            raise ValueError('More than one inference execution match specified name/model/version')
        inference_id = result[0]['inference_id']
    query_list = [{
        'field': 'source',
        'operator': 'EQ',
        'value': inference_id
    }]
    result = search_2d_poses(
        query_list=query_list,
        chunk_size=chunk_size
    )
    df = poses_2d_to_dataframe(result)
    return df

def fetch_2d_pose_data_by_time_span(
    environment_name,
    start_time,
    end_time=None,
    chunk_size=100,
    camera_device_types=['PI3WITHCAMERA'],
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    client = minimal_honeycomb.MinimalHoneycombClient(
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    query_list = list()
    logger.info('Finding camera assignments that match the specified environment, start time, and end time')
    result=client.bulk_query(
        request_name='findEnvironments',
        arguments={
            'name': {
                'type': 'String',
                'value': environment_name
            }
        },
        return_data=[
            'environment_id',
            {'assignments': [
                'assignment_id',
                'start',
                'end',
                {'assigned': [
                    {'... on Device': [
                        'device_id',
                        'device_type'
                    ]}
                ]}
            ]}
        ],
        id_field_name='environment_id'
    )
    if len(result) == 0:
        raise ValueError('No environments match name {}'.format(environment_name))
    if len(result) > 1:
        raise ValueError('More than one environment matches name {}'.format(environment_name))
    assignments = result[0].get('assignments')
    if assignments is None or len(assignments) == 0:
        raise ValueError('Environment {} has no assignments')
    camera_assignments = list()
    for assignment in assignments:
        if assignment.get('assigned', {}).get('device_type') in camera_device_types:
            camera_assignments.append(assignment)
    if len(camera_assignments) == 0:
        raise ValueError('No assignments in {} match device types {}'.format(
            environment_name,
            camera_device_types
        ))
    filtered_camera_assignments = minimal_honeycomb.filter_assignments(
        camera_assignments,
        start_time,
        end_time
    )
    if len(filtered_camera_assignments) == 0:
        raise ValueError('No camera assignments in {} match the specified start and end times'.format(
            environment_name
        ))
    camera_device_ids = [assignment.get('assigned').get('device_id') for assignment in filtered_camera_assignments]
    logger.info('Found {} camera assignments that match specified start and end times')
    query_list.append({
        'field': 'camera',
        'operator': 'IN',
        'values': camera_device_ids
    })
    if start_time is not None:
        query_list.append({
            'field': 'timestamp',
            'operator': 'GTE',
            'value': minimal_honeycomb.to_honeycomb_datetime(start_time)
        })
    if end_time is not None:
        query_list.append({
            'field': 'timestamp',
            'operator': 'LTE',
            'value': minimal_honeycomb.to_honeycomb_datetime(end_time)
        })
    result = search_2d_poses(
        query_list=query_list,
        chunk_size=chunk_size
    )
    df = poses_2d_to_dataframe(result)
    return df

def search_2d_poses(
    query_list,
    chunk_size=100,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    client = minimal_honeycomb.MinimalHoneycombClient(
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    logger.info('Searching for 2D poses that match the specified parameters')
    result = client.bulk_query(
        request_name='searchPoses2D',
        arguments={
            'query': {
                'type': 'QueryExpression!',
                'value': {
                    'operator': 'AND',
                    'children': query_list
                }
            }
        },
        return_data=[
            'pose_id',
            'timestamp',
            {'camera': [
                'device_id',
                'name'
            ]},
            'track_label',
            {'pose_model': [
                'pose_model_id'
            ]},
            {'keypoints': [
                'coordinates',
                'quality'
            ]},
            'quality',
            {'person': [
                'person_id'
            ]},
            {'source': [
                {'... on InferenceExecution': [
                    'inference_id'
                ]}
            ]}
        ],
        id_field_name = 'pose_id',
        chunk_size=chunk_size
    )
    logger.info('Fetched {} poses'.format(len(result)))
    return result

def poses_2d_to_dataframe(
    poses_2d
):
    pose_data = list()
    logger.info('Parsing {} poses'.format(len(poses_2d)))
    for pose in poses_2d:
        if pose.get('person') is None:
            pose['person'] = {}
        pose_data.append({
            'pose_id': pose.get('pose_id'),
            'timestamp': pose.get('timestamp'),
            'camera_device_id': pose.get('camera', {}).get('device_id'),
            'camera_name': pose.get('camera', {}).get('name'),
            'track_label': pose.get('track_label'),
            'pose_model_id': pose.get('pose_model', {}).get('pose_model_id'),
            'keypoint_array': np.array([keypoint.get('coordinates') for keypoint in pose.get('keypoints')]),
            'keypoint_quality_array': np.array([keypoint.get('quality') for keypoint in pose.get('keypoints')]),
            'pose_quality': pose.get('quality'),
            'person_id': pose.get('person', {}).get('person_id'),
            'inference_id': pose.get('source', {}).get('inference_id')
        })
    df = pd.DataFrame(pose_data)
    df.set_index('pose_id', inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.reindex(columns=[
        'camera_device_id',
        'camera_name',
        'track_label',
        'timestamp',
        'keypoint_array',
        'keypoint_quality_array',
        'pose_quality',
        'person_id',
        'inference_id',
        'pose_model_id'
    ])
    df.sort_values(['camera_name', 'camera_device_id', 'track_label', 'timestamp'], inplace=True)
    return df
