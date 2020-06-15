import pandas as pd
import numpy as np
import logging
from uuid import uuid4
import os
import re
import json

logger = logging.getLogger(__name__)

def fetch_2d_pose_data_from_local_json(
    directory_path
):
    data = list()
    for directory_entry in os.listdir(directory_path):
        if re.match(r'.*\.json', directory_entry):
            logger.info('Retrieving pose data from {}'.format(directory_entry))
            with open(os.path.join(directory_path, directory_entry), 'r') as fh:
                data_this_file = json.load(fh)
            logger.info('Retrieved {} poses from {}'.format(
                len(data_this_file),
                directory_entry
            ))
            data.extend(data_this_file)
    logger.info('Retrieved {} poses overall. Parsing')
    parsed_data = list()
    for datum in data:
        parsed_data.append({
            'pose_id_2d': uuid4().hex,
            'timestamp': datum.get('timestamp'),
            'camera_id': datum.get('camera'),
            'track_label_2d': datum.get('track_label'),
            'pose_model_id': datum.get('pose_model'),
            'keypoint_coordinates_2d': np.asarray([keypoint.get('coordinates') for keypoint in datum.get('keypoints')]),
            'keypoint_quality_2d': np.asarray([keypoint.get('quality') for keypoint in datum.get('keypoints')]),
            'pose_quality_2d': datum.get('quality')
        })
    poses_2d_df = pd.DataFrame(parsed_data)
    poses_2d_df['timestamp'] = pd.to_datetime(poses_2d_df['timestamp'])
    if poses_2d_df['pose_model_id'].nunique() > 1:
        raise ValueError('Returned poses are associated with multiple pose models')
    poses_2d_df.set_index('pose_id_2d', inplace=True)
    return poses_2d_df
