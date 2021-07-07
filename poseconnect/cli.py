import poseconnect.reconstruct
import poseconnect.track
import poseconnect.utils
import click

@click.group(
    help='Tools for constructing 3D pose tracks from multi-camera 2D poses'
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
        readable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.argument(
    'camera-calibrations-path',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.argument(
    'output-path',
    type=click.Path(
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.option(
    '--pose-model-name',
    type=click.Choice(
        list(poseconnect.reconstruct.KEYPOINT_CATEGORIES_BY_POSE_MODEL.keys()),
        case_sensitive=False
    ),
    help='Name of pose model'
)
@click.option(
    '--pose-3d-limits',
    type=click.STRING,
    default=None,
    help='Spatial limits for each keypoint (JSON-encoded array)',
)
@click.option(
    '--room-x-limits',
    nargs=2,
    type=click.FLOAT,
    default=None,
    help='Minimum and maximum x values of room (e.g., \"0.0 4.0\")'
)
@click.option(
    '--room-y-limits',
    nargs=2,
    type=click.FLOAT,
    default=None,
    help='Minimum and maximum y values of room (e.g., \"0.0 8.0\")'
)
@click.option(
    '--min-keypoint-quality',
    type=click.FLOAT,
    default=poseconnect.defaults.RECONSTRUCTION_MIN_KEYPOINT_QUALITY,
    help='Minimum 2D keypoint quality value for keypoint to be included in pose reconstruction',
    show_default=True
)
@click.option(
    '--min-num-keypoints',
    type=click.INT,
    default=poseconnect.defaults.RECONSTRUCTION_MIN_NUM_KEYPOINTS,
    help='Minimum number of valid keypoints for 2D pose to be included in pose reconstruction',
    show_default=True
)
@click.option(
    '--min-pose-quality',
    type=click.FLOAT,
    default=poseconnect.defaults.RECONSTRUCTION_MIN_POSE_QUALITY,
    help='Minimum pose quality for 2D pose to be included in pose reconstruction',
    show_default=True
)
@click.option(
    '--min-pose-pair-score',
    type=click.FLOAT,
    default=poseconnect.defaults.RECONSTRUCTION_MIN_POSE_PAIR_SCORE,
    help='Minimum 2D pose pair score for 3D pose to be included in pose reconstruction',
    show_default=True
)
@click.option(
    '--max-pose-pair-score',
    type=click.FLOAT,
    default=poseconnect.defaults.RECONSTRUCTION_MAX_POSE_PAIR_SCORE,
    help='Maximum 2D pose pair score for 3D pose to be included in pose reconstruction',
    show_default=True
)
@click.option(
    '--pose-pair-score-distance-method',
    type=click.Choice(
        ['pixels', 'probability'],
        case_sensitive=False
    ),
    default=poseconnect.defaults.RECONSTRUCTION_POSE_PAIR_SCORE_DISTANCE_METHOD,
    help='Method of determining 2D keypoint distance in reprojection score',
    show_default=True
)
@click.option(
    '--pose-pair-score-pixel-distance-scale',
    type=click.FLOAT,
    default=poseconnect.defaults.RECONSTRUCTION_POSE_PAIR_SCORE_PIXEL_DISTANCE_SCALE,
    help='Pixel distance scale (for \'probability\' method)',
    show_default=True
)
@click.option(
    '--pose-pair-score-summary-method',
    type=click.Choice(
        ['rms', 'sum'],
        case_sensitive=False
    ),
    default=poseconnect.defaults.RECONSTRUCTION_POSE_PAIR_SCORE_SUMMARY_METHOD,
    help='Method for summarizing reprojection distance over keypoints',
    show_default=True
)
@click.option(
    '--pose-3d-graph-initial-edge-threshold',
    type=click.INT,
    default=poseconnect.defaults.RECONSTRUCTION_POSE_3D_GRAPH_INITIAL_EDGE_THRESHOLD,
    help='Initial k value for splitting 3D pose graph into components',
    show_default=True
)
@click.option(
    '--pose_3d_graph_max_dispersion',
    type=click.FLOAT,
    default=poseconnect.defaults.RECONSTRUCTION_POSE_3D_GRAPH_MAX_DISPERSION,
    help='Maximum pose spatial dispersion allowed before further splitting 3D pose graph component',
    show_default=True
)
@click.option(
    '--include-track-labels/--exclude-track-labels',
    default=poseconnect.defaults.RECONSTRUCTION_INCLUDE_TRACK_LABELS,
    help='Include 2D pose track labels in 3D pose data',
    show_default=True
)
@click.option(
    '--progress-bar/--no-progress-bar',
    default=poseconnect.defaults.PROGRESS_BAR,
    required=False,
    help='Display progress bar',
    show_default=True
)
def cli_reconstruct_poses_3d(
    poses_2d_path,
    camera_calibrations_path,
    output_path,
    pose_3d_limits,
    pose_model_name,
    room_x_limits,
    room_y_limits,
    min_keypoint_quality,
    min_num_keypoints,
    min_pose_quality,
    min_pose_pair_score,
    max_pose_pair_score,
    pose_pair_score_distance_method,
    pose_pair_score_pixel_distance_scale,
    pose_pair_score_summary_method,
    pose_3d_graph_initial_edge_threshold,
    pose_3d_graph_max_dispersion,
    include_track_labels,
    progress_bar
):
    poses_3d = poseconnect.reconstruct.reconstruct_poses_3d(
        poses_2d=poses_2d_path,
        camera_calibrations=camera_calibrations_path,
        pose_3d_limits=pose_3d_limits,
        pose_model_name=pose_model_name,
        room_x_limits=room_x_limits,
        room_y_limits=room_y_limits,
        min_keypoint_quality=min_keypoint_quality,
        min_num_keypoints=min_num_keypoints,
        min_pose_quality=min_pose_quality,
        min_pose_pair_score=min_pose_pair_score,
        max_pose_pair_score=max_pose_pair_score,
        pose_pair_score_distance_method=pose_pair_score_distance_method,
        pose_pair_score_pixel_distance_scale=pose_pair_score_pixel_distance_scale,
        pose_pair_score_summary_method=pose_pair_score_summary_method,
        pose_3d_graph_initial_edge_threshold=pose_3d_graph_initial_edge_threshold,
        pose_3d_graph_max_dispersion=pose_3d_graph_max_dispersion,
        include_track_labels=include_track_labels,
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
    '--max-match-distance',
    type=click.FLOAT,
    default=poseconnect.defaults.TRACKING_MAX_MATCH_DISTANCE,
    help='Max distance allowed for matching pose with pose track',
    show_default=True
)
@click.option(
    '--max-iterations-since-last-match',
    type=click.INT,
    default=poseconnect.defaults.TRACKING_MAX_ITERATIONS_SINCE_LAST_MATCH,
    help='Maxmium number of iterations without match before ending pose track',
    show_default=True
)
@click.option(
    '--centroid-position-initial-sd',
    type=click.FLOAT,
    default=poseconnect.defaults.TRACKING_CENTROID_POSITION_INITIAL_SD,
    help='Initial position uncertainty for new pose track',
    show_default=True
)
@click.option(
    '--centroid-velocity-initial-sd',
    type=click.FLOAT,
    default=poseconnect.defaults.TRACKING_CENTROID_VELOCITY_INITIAL_SD,
    help='Initial velocity uncertainty for new pose track',
    show_default=True
)
@click.option(
    '--reference-delta-t-seconds',
    type=click.FLOAT,
    default=poseconnect.defaults.TRACKING_REFERENCE_DELTA_T_SECONDS,
    help='Time interval for velocity drift specification',
    show_default=True
)
@click.option(
    '--reference_velocity_drift',
    type=click.FLOAT,
    default=poseconnect.defaults.TRACKING_REFERENCE_VELOCITY_DRIFT,
    help='Velocity drift (std dev) over specified time interval',
    show_default=True
)
@click.option(
    '--position_observation_sd',
    type=click.FLOAT,
    default=poseconnect.defaults.TRACKING_POSITION_OBSERVATION_SD,
    help='Uncertainty in position observation',
    show_default=True
)
@click.option(
    '--num-poses-per-track-min',
    type=click.INT,
    default=poseconnect.defaults.TRACKING_NUM_POSES_PER_TRACK_MIN,
    help='Minimum number of poses in track',
    show_default=True
)
@click.option(
    '--progress-bar/--no-progress-bar',
    default=poseconnect.defaults.PROGRESS_BAR,
    required=False,
    help='Display progress bar',
    show_default=True
)
def cli_track_poses_3d(
    poses_3d_path,
    output_path,
    max_match_distance,
    max_iterations_since_last_match,
    centroid_position_initial_sd,
    centroid_velocity_initial_sd,
    reference_delta_t_seconds,
    reference_velocity_drift,
    position_observation_sd,
    num_poses_per_track_min,
    progress_bar
):
    poses_3d_with_tracks = poseconnect.track.track_poses_3d(
        poses_3d=poses_3d_path,
        max_match_distance=max_match_distance,
        max_iterations_since_last_match=max_iterations_since_last_match,
        centroid_position_initial_sd=centroid_position_initial_sd,
        centroid_velocity_initial_sd=centroid_velocity_initial_sd,
        reference_delta_t_seconds=reference_delta_t_seconds,
        reference_velocity_drift=reference_velocity_drift,
        position_observation_sd=position_observation_sd,
        num_poses_per_track_min=num_poses_per_track_min,
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
        readable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.argument(
    'output-path',
    type=click.Path(
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.option(
    '--frames-per-second',
    type=click.INT,
    default=poseconnect.defaults.FRAMES_PER_SECOND,
    help='Frames per second in 2D pose data',
    show_default=True
)
def cli_interpolate_pose_tracks_3d(
    poses_3d_with_tracks_path,
    frames_per_second,
    output_path
):
    poses_3d_with_tracks_interpolated = poseconnect.track.interpolate_pose_tracks_3d(
        poses_3d_with_tracks=poses_3d_with_tracks_path,
        frames_per_second=frames_per_second
    )
    poseconnect.utils.output_poses_3d_with_tracks(poses_3d_with_tracks_interpolated, output_path)

@click.command(
    name='identify',
    help='Identify 3D pose tracks using sensor data'
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
    'sensor-data-path',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.argument(
    'output-path',
    type=click.Path(
        resolve_path=True,
        allow_dash=False,
        path_type=None
    )
)
@click.option(
    '--frames-per-second',
    type=click.INT,
    default=poseconnect.defaults.FRAMES_PER_SECOND,
    help='Frames per second in 2D pose data',
    show_default=True
)
@click.option(
    '--sensor-position-keypoint-index',
    multiple=True,
    nargs=2,
    type=(str, int),
    help='Sensor position on each person (PERSON_ID KEYPOINT_INDEX)'
)
@click.option(
    '--active-person-id',
    multiple=True,
    help='Active person IDs (other IDs will be ignored)'
)
@click.option(
    '--ignore-z/--include-z',
    default=poseconnect.defaults.IDENTIFICATION_IGNORE_Z,
    help='Ignore z-axis position when comparing poses to sensor positions',
    show_default=True
)
@click.option(
    '--max-distance',
    type=click.FLOAT,
    default=poseconnect.defaults.IDENTIFICATION_MAX_DISTANCE,
    help='Maximum allowed distance between pose and sensor position for match',
    show_default=True
)
@click.option(
    '--min-fraction-matched',
    type=click.FLOAT,
    default=poseconnect.defaults.IDENTIFICATION_MIN_TRACK_FRACTION_MATCHED,
    help='Minimum fraction of pose track which must be matched to match track',
    show_default=True
)
def cli_identify_pose_tracks_3d(
    poses_3d_with_tracks_path,
    sensor_data_path,
    output_path,
    frames_per_second,
    sensor_position_keypoint_index,
    active_person_id,
    ignore_z,
    max_distance,
    min_fraction_matched
):
    sensor_position_keypoint_index = dict(sensor_position_keypoint_index)
    poses_3d_with_person_ids = poseconnect.identify.identify_pose_tracks_3d(
        poses_3d_with_tracks=poses_3d_with_tracks_path,
        sensor_data=sensor_data_path,
        frames_per_second=frames_per_second,
        id_field_names=poseconnect.defaults.IDENTIFICATION_ID_FIELD_NAMES,
        interpolation_field_names=poseconnect.defaults.IDENTIFICATION_INTERPOLATION_FIELD_NAMES,
        timestamp_field_name=poseconnect.defaults.IDENTIFICATION_TIMESTAMP_FIELD_NAME,
        sensor_position_keypoint_index=sensor_position_keypoint_index,
        active_person_ids=active_person_id,
        ignore_z=ignore_z,
        max_distance=max_distance,
        min_fraction_matched=min_fraction_matched
    )
    poseconnect.utils.output_poses_3d_with_tracks(poses_3d_with_person_ids, output_path)

cli.add_command(cli_reconstruct_poses_3d)
cli.add_command(cli_track_poses_3d)
cli.add_command(cli_interpolate_pose_tracks_3d)
cli.add_command(cli_identify_pose_tracks_3d)
