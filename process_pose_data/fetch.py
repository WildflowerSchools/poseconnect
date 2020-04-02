import minimal_honeycomb
import pandas as pd
import numpy as np
import tqdm
import datetime
import logging

logger = logging.getLogger(__name__)

def fetch_2d_pose_data_by_inference_execution(
    inference_id=None,
    inference_name=None,
    inference_model=None,
    inference_version=None,
    chunk_size=100,
    progress_bar=False,
    notebook=False,
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
    df = poses_2d_to_dataframe(
        poses_2d=result,
        progress_bar=progress_bar,
        notebook=notebook
    )
    return df

def fetch_2d_pose_data_by_time_span(
    environment_name,
    start_time,
    end_time=None,
    chunk_size=100,
    progress_bar=False,
    notebook=False,
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
    logger.info('Found {} camera assignments that match specified start and end times'.format(
        len(camera_device_ids)
    ))
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
    df = poses_2d_to_dataframe(
        poses_2d=result,
        progress_bar=progress_bar,
        notebook=notebook
    )
    return df

def fetch_2d_pose_data(
    start=None,
    end=None,
    environment_id=None,
    environment_name=None,
    camera_ids=None,
    camera_device_types=None,
    camera_part_numbers=None,
    camera_names=None,
    camera_serial_numbers=None,
    pose_model_id=None,
    pose_model_name=None,
    pose_model_variant_name=None,
    inference_ids=None,
    inference_names=None,
    inference_models=None,
    inference_versions=None,
    return_track_label=False,
    return_person_id=False,
    return_inference_id=False,
    return_pose_model_id=False,
    return_pose_quality=False,
    chunk_size=100,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    camera_ids_from_environment = fetch_camera_ids_from_environment(
        start=start,
        end=end,
        environment_id=environment_id,
        environment_name=environment_name,
        camera_device_types=camera_device_types,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    camera_ids_from_camera_properties = fetch_camera_ids_from_camera_properties(
        camera_ids=camera_ids,
        camera_device_types=camera_device_types,
        camera_part_numbers=camera_part_numbers,
        camera_names=camera_names,
        camera_serial_numbers=camera_serial_numbers,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    pose_model_id = fetch_pose_model_id(
        pose_model_id=pose_model_id,
        pose_model_name=pose_model_name,
        pose_model_variant_name=pose_model_variant_name,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    inference_ids = fetch_inference_ids(
        inference_ids=inference_ids,
        inference_names=inference_names,
        inference_models=inference_models,
        inference_versions=inference_versions,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    logger.info('Building query list for 2D pose search')
    query_list = list()
    if start is not None:
        query_list.append({
            'field': 'timestamp',
            'operator': 'GTE',
            'value': minimal_honeycomb.to_honeycomb_datetime(start)
        })
    if end is not None:
        query_list.append({
            'field': 'timestamp',
            'operator': 'LTE',
            'value': minimal_honeycomb.to_honeycomb_datetime(end)
        })
    if camera_ids_from_environment is not None:
        query_list.append({
            'field': 'camera',
            'operator': 'IN',
            'values': camera_ids_from_environment
        })
    if camera_ids_from_camera_properties is not None:
        query_list.append({
            'field': 'camera',
            'operator': 'IN',
            'values': camera_ids_from_camera_properties
        })
    if pose_model_id is not None:
        query_list.append({
            'field': 'pose_model',
            'operator': 'EQ',
            'value': pose_model_id
        })
    if inference_ids is not None:
        query_list.append({
            'field': 'source',
            'operator': 'IN',
            'values': inference_ids
        })
    return_data= [
        'pose_id',
        'timestamp',
        {'camera': [
            'device_id'
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
    ]
    result = search_2d_poses(
        query_list=query_list,
        return_data=return_data,
        chunk_size=chunk_size,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    data = list()
    logger.info('Parsing {} returned poses'.format(len(result)))
    for datum in result:
        data.append({
            'pose_id': datum.get('pose_id'),
            'timestamp': datum.get('timestamp'),
            'camera_id': (datum.get('camera') if datum.get('camera') is not None else {}).get('device_id'),
            'track_label': datum.get('track_label'),
            'person_id': (datum.get('person') if datum.get('person') is not None else {}).get('person_id'),
            'inference_id': (datum.get('source') if datum.get('source') is not None else {}).get('inference_id'),
            'pose_model_id': (datum.get('pose_model') if datum.get('pose_model') is not None else {}).get('pose_model_id'),
            'keypoint_coordinates': np.asarray([keypoint.get('coordinates') for keypoint in datum.get('keypoints')]),
            'keypoint_quality': np.asarray([keypoint.get('quality') for keypoint in datum.get('keypoints')]),
            'pose_quality': datum.get('quality')
        })
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['pose_model_id'].nunique() > 1:
        raise ValueError('Returned poses are associated with multiple pose models')
    if (df.groupby(['timestamp', 'camera_id'])['inference_id'].nunique() > 1).any():
        raise ValueError('Returned poses have multiple inference IDs for some camera IDs at some timestamps')
    df.set_index('pose_id', inplace=True)
    return_columns = [
        'timestamp',
        'camera_id'
    ]
    if return_track_label:
        return_columns.append('track_label')
    if return_person_id:
        return_columns.append('person_id')
    if return_inference_id:
        return_columns.append('inference_id')
    if return_pose_model_id:
        return_columns.append('pose_model_id')
    return_columns.extend([
        'keypoint_coordinates',
        'keypoint_quality'
    ])
    if return_pose_quality:
        return_columns.append('pose_quality')
    df = df.reindex(columns=return_columns)
    return df

def fetch_camera_ids_from_environment(
    start=None,
    end=None,
    environment_id=None,
    environment_name=None,
    camera_device_types=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    if camera_device_types is None:
        camera_device_types = [
            'PI3WITHCAMERA',
            'PIZEROWITHCAMERA'
        ]
    environment_id = fetch_environment_id(
        environment_id=environment_id,
        environment_name=environment_name,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    if environment_id is None:
        return None
    logger.info('Fetching camera assignments for specified environment and time span')
    client = minimal_honeycomb.MinimalHoneycombClient(
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    result = client.request(
        request_type='query',
        request_name='getEnvironment',
        arguments={
            'environment_id': {
                'type': 'ID!',
                'value': environment_id
            }
        },
        return_object=[
            {'assignments': [
                'start',
                'end',
                {'assigned': [
                    {'... on Device': [
                        'device_id',
                        'device_type'
                    ]}
                ]}
            ]}
        ]
    )
    filtered_assignments = minimal_honeycomb.filter_assignments(
        assignments=result.get('assignments'),
        start_time=start,
        end_time=end
    )
    camera_device_ids = list()
    for assignment in filtered_assignments:
        device_type = assignment.get('assigned').get('device_type')
        if device_type is not None and device_type in camera_device_types:
            camera_device_ids.append(assignment.get('assigned').get('device_id'))
    if len(camera_device_ids) == 0:
        raise ValueError('No camera devices found in specified environment for specified time span')
    logger.info('Found {} camera assignments for specified environment and time span'.format(len(camera_device_ids)))
    return camera_device_ids

def fetch_environment_id(
    environment_id=None,
    environment_name=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    if environment_id is not None:
        if environment_name is not None:
            raise ValueError('If environment ID is specified, environment name cannot be specified')
        return environment_id
    if environment_name is not None:
        logger.info('Fetching environment ID for specified environment name')
        client = minimal_honeycomb.MinimalHoneycombClient(
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
        result = client.bulk_query(
            request_name='findEnvironments',
            arguments={
                'name': {
                    'type': 'String',
                    'value': environment_name
                }
            },
            return_data=[
                'environment_id'
            ],
            id_field_name='environment_id'
        )
        if len(result) == 0:
            raise ValueError('No environments match environment name {}'.format(
                environment_name
            ))
        if len(result) > 1:
            raise ValueError('Multiple environments match environment name {}'.format(
                environment_name
            ))
        environment_id = result[0].get('environment_id')
        logger.info('Found environment ID for specified environment name')
        return environment_id
    return None

def fetch_camera_ids_from_camera_properties(
    camera_ids=None,
    camera_device_types=None,
    camera_part_numbers=None,
    camera_names=None,
    camera_serial_numbers=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    if camera_ids is not None:
        if camera_names is not None or camera_part_numbers is not None or camera_serial_numbers is not None:
            raise ValueError('If camera IDs are specified, camera names/part numbers/serial numbers cannot be specified')
        return camera_ids
    if camera_names is not None or camera_part_numbers is not None or camera_serial_numbers is not None:
        query_list=list()
        if camera_device_types is not None:
            query_list.append({
                'field': 'device_type',
                'operator': 'IN',
                'values': camera_device_types
            })
        if camera_part_numbers is not None:
            query_list.append({
                'field': 'part_number',
                'operator': 'IN',
                'values': camera_part_numbers
            })
        if camera_names is not None:
            query_list.append({
                'field': 'name',
                'operator': 'IN',
                'values': camera_names
            })
        if camera_serial_numbers is not None:
            query_list.append({
                'field': 'serial_number',
                'operator': 'IN',
                'values': camera_serial_numbers
            })
        logger.info('Fetching camera IDs for cameras with specified properties')
        client = minimal_honeycomb.MinimalHoneycombClient(
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
        result = client.bulk_query(
            request_name='searchDevices',
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
                'device_id'
            ],
            id_field_name='device_id'
        )
        if len(result) == 0:
            raise ValueError('No devices match specified device types/part numbers/names/serial numbers')
        camera_ids = [datum.get('device_id') for datum in result]
        logger.info('Found {} camera IDs that match specified properties'.format(len(camera_ids)))
        return camera_ids
    return None

def fetch_pose_model_id(
    pose_model_id=None,
    pose_model_name=None,
    pose_model_variant_name=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    if pose_model_id is not None:
        if pose_model_name is not None or pose_model_variant_name is not None:
            raise ValueError('If pose model ID is specified, pose model name/variant name cannot be specified')
        return pose_model_id
    if pose_model_name is not None or pose_model_variant_name is not None:
        arguments=dict()
        if pose_model_name is not None:
            arguments['model_name'] = {
                'type': 'String',
                'value': pose_model_name
            }
        if pose_model_variant_name is not None:
            arguments['model_variant_name'] = {
                'type': 'String',
                'value': pose_model_variant_name
            }
        logger.info('Fetching pose model ID for pose model with specified properties')
        client = minimal_honeycomb.MinimalHoneycombClient(
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
        result = client.bulk_query(
            request_name='findPoseModels',
            arguments=arguments,
            return_data=[
                'pose_model_id'
            ],
            id_field_name='pose_model_id'
        )
        if len(result) == 0:
            raise ValueError('No pose models match specified model name/model variant name')
        if len(result) > 1:
            raise ValueError('Multiple pose models match specified model name/model variant name')
        pose_model_id = result[0].get('pose_model_id')
        logger.info('Found pose model ID for pose model with specified properties')
        return pose_model_id
    return None

def fetch_inference_ids(
    inference_ids=None,
    inference_names=None,
    inference_models=None,
    inference_versions=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    if inference_ids is not None:
        if inference_names is not None or inference_models is not None or inference_versions is not None:
            raise ValueError('If inference IDs are specified, inference names/models/versions cannot be specified')
        return inference_ids
    if inference_names is not None or inference_models is not None or inference_versions is not None:
        query_list=list()
        if inference_names is not None:
            query_list.append({
                'field': 'name',
                'operator': 'IN',
                'values': inference_names
            })
        if inference_models is not None:
            query_list.append({
                'field': 'model',
                'operator': 'IN',
                'values': inference_models
            })
        if inference_versions is not None:
            query_list.append({
                'field': 'version',
                'operator': 'IN',
                'values': inference_versions
            })
        logger.info('Fetching inference IDs for inference runs with specified properties')
        client = minimal_honeycomb.MinimalHoneycombClient(
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
        result = client.bulk_query(
            request_name='searchInferenceExecutions',
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
                'inference_id'
            ],
            id_field_name='inference_id'
        )
        if len(result) == 0:
            raise ValueError('No inference executions match specified inference names/models/versions')
        inference_ids = [datum.get('inference_id') for datum in result]
        logger.info('Found {} inference runs that match specified properties'.format(len(inference_ids)))
        return inference_ids
    return None

def search_2d_poses(
    query_list,
    return_data,
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
        return_data=return_data,
        id_field_name = 'pose_id',
        chunk_size=chunk_size
    )
    logger.info('Fetched {} poses'.format(len(result)))
    return result

def poses_2d_to_dataframe(
    poses_2d,
    progress_bar=False,
    notebook=False
):
    pose_data = list()
    logger.info('Parsing {} poses'.format(len(poses_2d)))
    if progress_bar:
        if notebook:
            poses_2d=tqdm.tqdm_notebook(poses_2d)
        else:
            poses_2d=tqdm.tqdm(poses_2d)
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
    df['centroid'] = df['keypoint_array'].apply(lambda x: np.nanmean(x, axis=0))
    df = df.reindex(columns=[
        'camera_device_id',
        'camera_name',
        'track_label',
        'timestamp',
        'keypoint_array',
        'centroid',
        'keypoint_quality_array',
        'pose_quality',
        'person_id',
        'inference_id',
        'pose_model_id'
    ])
    df.sort_values(['camera_name', 'camera_device_id', 'track_label', 'timestamp'], inplace=True)
    return df

def extract_pose_model_id(
    df
):
    pose_model_ids = df['pose_model_id'].unique().tolist()
    if len(pose_model_ids) > 1:
        raise ValueError('Data frame contains multiple pose model ids: {}'.format(
            pose_model_ids
        ))
    pose_model_id = pose_model_ids[0]
    logger.info('Data conforms to pose model {}'.format(
        pose_model_id
    ))
    return pose_model_id

def fetch_pose_model_info(
    pose_model_id,
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
    logger.info('Fetching info for pose model {}'.format(
        pose_model_id
    ))
    result = client.request(
        request_type='query',
        request_name='getPoseModel',
        arguments={
            'pose_model_id': {
                'type': 'ID!',
                'value': pose_model_id
            }
        },
        return_object=[
            'model_name',
            'model_variant_name',
            'keypoint_names',
            'keypoint_descriptions',
            'keypoint_connectors'
        ]
    )
    return result

def fetch_camera_assignment_ids(
    camera_device_ids,
    start_time,
    end_time,
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
    logger.info('Fetching assignment IDs for {} cameras'.format(len(camera_device_ids)))
    result = client.bulk_query(
        request_name='searchDevices',
        arguments={
            'query': {
                'type': 'QueryExpression!',
                'value': {
                    'field': 'device_id',
                    'operator': 'IN',
                    'values': camera_device_ids
                }
            }
        },
        return_data=[
            'device_id',
            {'assignments': [
                'assignment_id',
                'start',
                'end'
            ]}
        ],
        id_field_name='device_id'
    )
    assignment_id_lookup = dict()
    for device in result:
        assignment = minimal_honeycomb.extract_assignment(
            device['assignments'],
            start_time=start_time,
            end_time=end_time
        )
        assignment_id_lookup[device['device_id']] = assignment['assignment_id']
    return assignment_id_lookup
