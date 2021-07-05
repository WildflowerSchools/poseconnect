import poseconnect.reconstruct
import poseconnect.utils
import click

@click.group(
    help='Process and transform human pose data'
)
def cli():
    pass

@click.command(
    name='reconstruct',
    help='Reconstruct 3D poses from 2D poses and camera calibration info'
)
@click.argument(
    'poses-2d-path',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=True,
        writable=False,
        readable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.argument(
    'camera-calibrations-path',
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=True,
        writable=False,
        readable=False,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.argument(
    'output-path',
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=True,
        writable=False,
        readable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.option(
    '--pose-model-name',
    required=True,
    type=click.STRING,
    help='Name of pose model (e.g., \"COCO-17\")'
)
@click.option(
    '--room-x-limits',
    required=True,
    nargs=2,
    type=click.FLOAT,
    help='Minimum and maximum x values of room (e.g., \"0.0 4.0\")'
)
@click.option(
    '--room-y-limits',
    required=True,
    nargs=2,
    type=click.FLOAT,
    help='Minimum and maximum y values of room (e.g., \"0.0 8.0\")'
)
@click.option(
    '--progress-bar/--no-progress-bar',
    default=False,
    required=False,
    help='Display progress bar',
    show_default=True
)
def cli_reconstruct_poses_3d(
    poses_2d_path,
    camera_calibrations_path,
    output_path,
    pose_model_name,
    room_x_limits,
    room_y_limits,
    progress_bar
):
    poses_3d = poseconnect.reconstruct.reconstruct_poses_3d(
        poses_2d=poses_2d_path,
        camera_calibrations=camera_calibrations_path,
        pose_3d_limits=None,
        pose_model_name=pose_model_name,
        room_x_limits=room_x_limits,
        room_y_limits=room_y_limits,
        min_keypoint_quality=poseconnect.defaults.RECONSTRUCTION_MIN_KEYPOINT_QUALITY,
        min_num_keypoints=poseconnect.defaults.RECONSTRUCTION_MIN_NUM_KEYPOINTS,
        min_pose_quality=poseconnect.defaults.RECONSTRUCTION_MIN_POSE_QUALITY,
        min_pose_pair_score=poseconnect.defaults.RECONSTRUCTION_MIN_POSE_PAIR_SCORE,
        max_pose_pair_score=poseconnect.defaults.RECONSTRUCTION_MAX_POSE_PAIR_SCORE,
        pose_pair_score_distance_method=poseconnect.defaults.RECONSTRUCTION_POSE_PAIR_SCORE_DISTANCE_METHOD,
        pose_pair_score_pixel_distance_scale=poseconnect.defaults.RECONSTRUCTION_POSE_PAIR_SCORE_PIXEL_DISTANCE_SCALE,
        pose_pair_score_summary_method=poseconnect.defaults.RECONSTRUCTION_POSE_PAIR_SCORE_SUMMARY_METHOD,
        pose_3d_graph_initial_edge_threshold=poseconnect.defaults.RECONSTRUCTION_POSE_3D_GRAPH_INITIAL_EDGE_THRESHOLD,
        pose_3d_graph_max_dispersion=poseconnect.defaults.RECONSTRUCTION_POSE_3D_GRAPH_MAX_DISPERSION,
        include_track_labels=poseconnect.defaults.RECONSTRUCTION_INCLUDE_TRACK_LABELS,
        progress_bar=progress_bar,
        notebook=False
    )
    poseconnect.utils.output_poses_3d(poses_3d, output_path)

cli.add_command(cli_reconstruct_poses_3d)
