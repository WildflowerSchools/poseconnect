import poseconnect.reconstruct
import poseconnect.track
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
        dir_okay=False,
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
        dir_okay=False,
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

@click.command(
    name='track',
    help='Connect 3D poses into 3D pose tracks'
)
@click.argument(
    'poses-3d-path',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.argument(
    'output-path',
    type=click.Path(
        exists=False,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.option(
    '--progress-bar/--no-progress-bar',
    default=False,
    required=False,
    help='Display progress bar',
    show_default=True
)
def cli_track_poses_3d(
    poses_3d_path,
    output_path,
    progress_bar
):
    poses_3d_with_tracks = poseconnect.track.track_poses_3d(
        poses_3d=poses_3d_path,
        max_match_distance=poseconnect.defaults.TRACKING_MAX_MATCH_DISTANCE,
        max_iterations_since_last_match=poseconnect.defaults.TRACKING_MAX_ITERATIONS_SINCE_LAST_MATCH,
        centroid_position_initial_sd=poseconnect.defaults.TRACKING_CENTROID_POSITION_INITIAL_SD,
        centroid_velocity_initial_sd=poseconnect.defaults.TRACKING_CENTROID_VELOCITY_INITIAL_SD,
        reference_delta_t_seconds=poseconnect.defaults.TRACKING_REFERENCE_DELTA_T_SECONDS,
        reference_velocity_drift=poseconnect.defaults.TRACKING_REFERENCE_VELOCITY_DRIFT,
        position_observation_sd=poseconnect.defaults.TRACKING_POSITION_OBSERVATION_SD,
        num_poses_per_track_min=poseconnect.defaults.TRACKING_NUM_POSES_PER_TRACK_MIN,
        progress_bar=progress_bar,
        notebook=False
    )
    poseconnect.utils.output_poses_3d_with_tracks(poses_3d_with_tracks, output_path)

@click.command(
    name='interpolate',
    help='Interpolate to fill gaps in 3D pose tracks'
)
@click.argument(
    'poses-3d-with-tracks-path',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.argument(
    'output-path',
    type=click.Path(
        exists=False,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
def cli_interpolate_pose_tracks_3d(
    poses_3d_with_tracks_path,
    output_path
):
    poses_3d_with_tracks_interpolated = poseconnect.track.interpolate_pose_tracks_3d(
        poses_3d_with_tracks=poses_3d_with_tracks_path,
        frames_per_second=poseconnect.defaults.FRAMES_PER_SECOND
    )
    poseconnect.utils.output_poses_3d_with_tracks(poses_3d_with_tracks_interpolated, output_path)

cli.add_command(cli_reconstruct_poses_3d)
cli.add_command(cli_track_poses_3d)
cli.add_command(cli_interpolate_pose_tracks_3d)
