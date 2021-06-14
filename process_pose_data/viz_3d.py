from collections import OrderedDict
import json
import os

def convert_3d_poses_with_person_info_to_json(
    poses_3d_with_person_info_df,
    output_path=None
):
    poses_3d_with_person_info_df = poses_3d_with_person_info_df.copy()
    poses_3d_with_person_info_df.index.name = 'pose_3d_id'
    poses_3d_with_person_info_df.reset_index(inplace=True)
    poses_3d_with_person_info_df.sort_values('timestamp', inplace=True)
    poses_3d_with_person_info_df['timestamp'] = poses_3d_with_person_info_df['timestamp'].apply(lambda x: x.isoformat())
    poses_3d_with_person_info_df['keypoint_coordinates_3d'] = poses_3d_with_person_info_df['keypoint_coordinates_3d'].apply(lambda x: x.tolist())
    poses_3d_with_person_info_df = poses_3d_with_person_info_df.reindex(columns=[
        'timestamp',
        'pose_3d_id',
        'keypoint_coordinates_3d',
        'pose_track_3d_id',
        'person_id',
        'name',
        'short_name'
    ])
    data_dict = OrderedDict()
    for timestamp, timestamp_df in poses_3d_with_person_info_df.groupby('timestamp'):
        data_dict[timestamp] = timestamp_df.to_dict(orient='records')
    poses_3d_with_person_info_json = json.dumps(data_dict, indent=2)
    if output_path is not None:
        output_directory = os.path.dirname(output_path)
        os.makedirs(output_directory, exist_ok=True)
        with open(output_path, 'w') as fp:
            fp.write(poses_3d_with_person_info_json)
    return poses_3d_with_person_info_json
