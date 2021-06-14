import json
import os

def convert_3d_poses_with_person_info_to_json(
    poses_3d_with_person_info_df,
    output_path=None
):
    poses_3d_with_person_info_json = poses_3d_with_person_info_df.to_json(
        orient='index',
        date_format='iso'
    )
    if output_path is not None:
        output_directory = os.path.dirname(output_path)
        os.makedirs(output_directory, exist_ok=True)
        with open(output_path, 'w') as fp:
            fp.write(poses_3d_with_person_info_json)
    return poses_3d_with_person_info_json
