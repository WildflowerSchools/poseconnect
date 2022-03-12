import poseconnect.reconstruct
import poseconnect.track
import poseconnect.utils
import dateutil
import click
import logging

logger = logging.getLogger(__name__)

class TimezoneType(click.ParamType):
    name = "timezone"
    def convert(self, value, param, ctx):
        tzinfo = dateutil.tz.gettz(value)
        if tzinfo is None:
            self.fail('Timezone \'value\' not recognized', param, ctx)
        return tzinfo

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
    '--floor-z',
    type=click.FLOAT,
    default=poseconnect.defaults.POSE_3D_FLOOR_Z,
    help='Height (z-coordinate) of room floor (used for filtering 3D poses)',
    show_default=True
)
@click.option(
    '--foot-z-limits',
    nargs=2,
    type=click.FLOAT,
    default=poseconnect.defaults.POSE_3D_FOOT_Z_LIMITS,
    help='Minimum and maximum height above floor for feet in reconstructed 3D poses (e.g., \"0.0 1.0\")',
    show_default=False
)
@click.option(
    '--knee-z-limits',
    nargs=2,
    type=click.FLOAT,
    default=poseconnect.defaults.POSE_3D_KNEE_Z_LIMITS,
    help='Minimum and maximum height above floor for knees in reconstructed 3D poses (e.g., \"0.0 1.0\")',
    show_default=False
)
@click.option(
    '--hip-z-limits',
    nargs=2,
    type=click.FLOAT,
    default=poseconnect.defaults.POSE_3D_HIP_Z_LIMITS,
    help='Minimum and maximum height above floor for hips in reconstructed 3D poses (e.g., \"0.0 1.5\")',
    show_default=False
)
@click.option(
    '--thorax-z-limits',
    nargs=2,
    type=click.FLOAT,
    default=poseconnect.defaults.POSE_3D_THORAX_Z_LIMITS,
    help='Minimum and maximum height above floor for thoraxes in reconstructed 3D poses (e.g., \"0.0 1.7\")',
    show_default=False
)
@click.option(
    '--shoulder-z-limits',
    nargs=2,
    type=click.FLOAT,
    default=poseconnect.defaults.POSE_3D_SHOULDER_Z_LIMITS,
    help='Minimum and maximum height above floor for shoulders in reconstructed 3D poses (e.g., \"0.0 1.9\")',
    show_default=False
)
@click.option(
    '--elbow-z-limits',
    nargs=2,
    type=click.FLOAT,
    default=poseconnect.defaults.POSE_3D_ELBOW_Z_LIMITS,
    help='Minimum and maximum height above floor for elbows in reconstructed 3D poses (e.g., \"0.0 2.0\")',
    show_default=False
)
@click.option(
    '--hand-z-limits',
    nargs=2,
    type=click.FLOAT,
    default=poseconnect.defaults.POSE_3D_HAND_Z_LIMITS,
    help='Minimum and maximum height above floor for hands in reconstructed 3D poses (e.g., \"0.0 3.0\")',
    show_default=False
)
@click.option(
    '--neck-z-limits',
    nargs=2,
    type=click.FLOAT,
    default=poseconnect.defaults.POSE_3D_NECK_Z_LIMITS,
    help='Minimum and maximum height above floor for necks in reconstructed 3D poses (e.g., \"0.0 1.9\")',
    show_default=False
)
@click.option(
    '--head-z-limits',
    nargs=2,
    type=click.FLOAT,
    default=poseconnect.defaults.POSE_3D_HEAD_Z_LIMITS,
    help='Minimum and maximum height above floor for heads in reconstructed 3D poses (e.g., \"0.0 2.0\")',
    show_default=False
)
@click.option(
    '--tolerance',
    type=click.FLOAT,
    default=poseconnect.defaults.POSE_3D_LIMITS_TOLERANCE,
    help='Tolerance for 3D pose spatial limits',
    show_default=True
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
        ['pixels', 'image_frac', '3d'],
        case_sensitive=False
    ),
    default=poseconnect.defaults.RECONSTRUCTION_POSE_PAIR_SCORE_DISTANCE_METHOD,
    help='Method of determining 2D keypoint distance in reprojection score',
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
    '--parallel/--no-parallel',
    default=poseconnect.defaults.RECONSTRUCTION_PARALLEL,
    help='Distribute reconstruction across cores',
    show_default=True
)
@click.option(
    '--num-parallel-processes',
    type=click.INT,
    default=poseconnect.defaults.RECONSTRUCTION_NUM_PARALLEL_PROCESSES,
    help='Number of parallel processes to use',
    show_default=True
)
@click.option(
    '--num-chunks',
    type=click.INT,
    default=poseconnect.defaults.RECONSTRUCTION_NUM_CHUNKS,
    help='Number of chunks to separate 2D poses into',
    show_default=True
)
@click.option(
    '--progress-bar/--no-progress-bar',
    default=poseconnect.defaults.PROGRESS_BAR,
    required=False,
    help='Display progress bar',
    show_default=True
)
@click.option(
    '--log-level',
    type=click.Choice(
        poseconnect.defaults.LOG_LEVEL_OPTIONS,
        case_sensitive=False
    ),
    default=poseconnect.defaults.LOG_LEVEL,
    help='Log level',
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
    floor_z,
    foot_z_limits,
    knee_z_limits,
    hip_z_limits,
    thorax_z_limits,
    shoulder_z_limits,
    elbow_z_limits,
    hand_z_limits,
    neck_z_limits,
    head_z_limits,
    tolerance,
    min_keypoint_quality,
    min_num_keypoints,
    min_pose_quality,
    min_pose_pair_score,
    max_pose_pair_score,
    pose_pair_score_distance_method,
    pose_3d_graph_initial_edge_threshold,
    pose_3d_graph_max_dispersion,
    include_track_labels,
    parallel,
    num_parallel_processes,
    num_chunks,
    progress_bar,
    log_level
):
    if log_level is not None:
        numeric_log_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_log_level, int):
            raise ValueError('Invalid log level: %s'.format(log_level))
        logging.basicConfig(level=numeric_log_level)
    poses_3d = poseconnect.reconstruct.reconstruct_poses_3d(
        poses_2d=poses_2d_path,
        camera_calibrations=camera_calibrations_path,
        pose_3d_limits=pose_3d_limits,
        pose_model_name=pose_model_name,
        room_x_limits=room_x_limits,
        room_y_limits=room_y_limits,
        floor_z=floor_z,
        foot_z_limits=foot_z_limits,
        knee_z_limits=knee_z_limits,
        hip_z_limits=hip_z_limits,
        thorax_z_limits=thorax_z_limits,
        shoulder_z_limits=shoulder_z_limits,
        elbow_z_limits=elbow_z_limits,
        hand_z_limits=hand_z_limits,
        neck_z_limits=neck_z_limits,
        head_z_limits=head_z_limits,
        tolerance=tolerance,
        min_keypoint_quality=min_keypoint_quality,
        min_num_keypoints=min_num_keypoints,
        min_pose_quality=min_pose_quality,
        min_pose_pair_score=min_pose_pair_score,
        max_pose_pair_score=max_pose_pair_score,
        pose_pair_score_distance_method=pose_pair_score_distance_method,
        pose_3d_graph_initial_edge_threshold=pose_3d_graph_initial_edge_threshold,
        pose_3d_graph_max_dispersion=pose_3d_graph_max_dispersion,
        include_track_labels=include_track_labels,
        parallel=parallel,
        num_parallel_processes=num_parallel_processes,
        num_chunks=num_chunks,
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
@click.option(
    '--log-level',
    type=click.Choice(
        poseconnect.defaults.LOG_LEVEL_OPTIONS,
        case_sensitive=False
    ),
    default=poseconnect.defaults.LOG_LEVEL,
    help='Log level',
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
    progress_bar,
    log_level
):
    if log_level is not None:
        numeric_log_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_log_level, int):
            raise ValueError('Invalid log level: %s'.format(log_level))
        logging.basicConfig(level=numeric_log_level)
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
@click.option(
    '--log-level',
    type=click.Choice(
        poseconnect.defaults.LOG_LEVEL_OPTIONS,
        case_sensitive=False
    ),
    default=poseconnect.defaults.LOG_LEVEL,
    help='Log level',
    show_default=True
)
def cli_interpolate_pose_tracks_3d(
    poses_3d_with_tracks_path,
    frames_per_second,
    output_path,
    log_level
):
    if log_level is not None:
        numeric_log_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_log_level, int):
            raise ValueError('Invalid log level: %s'.format(log_level))
        logging.basicConfig(level=numeric_log_level)
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
@click.option(
    '--log-level',
    type=click.Choice(
        poseconnect.defaults.LOG_LEVEL_OPTIONS,
        case_sensitive=False
    ),
    default=poseconnect.defaults.LOG_LEVEL,
    help='Log level',
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
    min_fraction_matched,
    log_level
):
    if log_level is not None:
        numeric_log_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_log_level, int):
            raise ValueError('Invalid log level: %s'.format(log_level))
        logging.basicConfig(level=numeric_log_level)
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

@click.command(
    name='overlay',
    help='Overlay poses onto video'
)
@click.argument(
    'poses-path',
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
    'video-input-path',
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
    'video-start-time',
    type=click.DateTime(formats=None)
)
@click.option(
    '--video-start-timezone',
    type=TimezoneType(),
    default=poseconnect.defaults.OVERLAY_VIDEO_START_TIMEZONE,
    help='Timezone of video start time',
    show_default=True
)
@click.option(
    '--pose-type',
    type=click.Choice(
        ['2d', '3d'],
        case_sensitive=False
    ),
    default=poseconnect.defaults.OVERLAY_POSE_TYPE,
    help='Dimensionality of pose data',
    show_default=True
)
@click.option(
    '--camera-id',
    type=click.STRING,
    default=poseconnect.defaults.OVERLAY_CAMERA_ID,
    help='Camera ID associated with video',
    show_default=True
)
@click.option(
    '--camera-calibrations-path',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    ),
    default=None,
    help='Path to file containing camera calibration data',
    show_default=True
)
@click.option(
    '--pose-label-column',
    type=click.STRING,
    default=poseconnect.defaults.OVERLAY_POSE_LABEL_COLUMN,
    help='Name of field containing pose labels',
    show_default=True
)
@click.option(
    '--pose-label-map-path',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        allow_dash=False,
        path_type=None
    ),
    default=None,
    help='Path to file containing pose label mapping',
    show_default=True
)
@click.option(
    '--generate-pose-label-map/--preserve-pose-labels',
    default=poseconnect.defaults.OVERLAY_GENERATE_POSE_LABEL_MAP,
    help='Generate pose label map to shorten labels',
    show_default=False
)
@click.option(
    '--video-fps',
    type=click.FLOAT,
    default=poseconnect.defaults.OVERLAY_VIDEO_FPS,
    help='Video frame rate in frames per second',
    show_default=True
)
@click.option(
    '--video-frame-count',
    type=click.INT,
    default=poseconnect.defaults.OVERLAY_VIDEO_FRAME_COUNT,
    help='Number of frames in video',
    show_default=True
)
@click.option(
    '--video-output-path',
    type=click.Path(
        exists=False
    ),
    default=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_PATH,
    help='Path to video output file',
    show_default=True
)
@click.option(
    '--video-output-directory',
    type=click.Path(
        exists=False
    ),
    default=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_DIRECTORY,
    help='Path to directory containing video output files',
    show_default=True
)
@click.option(
    '--video-output-filename-suffix',
    type=click.STRING,
    default=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FILENAME_SUFFIX,
    help='String to append to video input filename to generate video output filename',
    show_default=True
)
@click.option(
    '--video-output-filename-extension',
    type=click.STRING,
    default=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FILENAME_EXTENSION,
    help='Filename extension for video output file (determines file type)',
    show_default=True
)
@click.option(
    '--video-output-fourcc-string',
    type=click.STRING,
    default=poseconnect.defaults.OVERLAY_VIDEO_OUTPUT_FOURCC_STRING,
    help='Video output codec specification (see fourcc.org)',
    show_default=True
)
@click.option(
    '--draw-keypoint-connectors/--omit-keypoint-connectors',
    default=poseconnect.defaults.OVERLAY_DRAW_KEYPOINT_CONNECTORS,
    help='Generate pose label map to shorten labels',
    show_default=False
)
@click.option(
    '--pose-model-name',
    type=click.Choice(
        list(poseconnect.reconstruct.KEYPOINT_CATEGORIES_BY_POSE_MODEL.keys()),
        case_sensitive=False
    ),
    help='Name of pose model',
    show_default=True
)
@click.option(
    '--pose-color',
    type=click.STRING,
    default=poseconnect.defaults.OVERLAY_POSE_COLOR,
    help='Color for keypoints and keypoint connectors',
    show_default=True
)
@click.option(
    '--keypoint-radius',
    type=click.FLOAT,
    default=poseconnect.defaults.OVERLAY_KEYPOINT_RADIUS,
    help='Radius of each keypoint in pixels',
    show_default=True
)
@click.option(
    '--keypoint-alpha',
    type=click.FLOAT,
    default=poseconnect.defaults.OVERLAY_KEYPOINT_ALPHA,
    help='Transparency of keypoints',
    show_default=True
)
@click.option(
    '--keypoint-connector-alpha',
    type=click.FLOAT,
    default=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_ALPHA,
    help='Transparency of keypoint connectors',
    show_default=True
)
@click.option(
    '--keypoint-connector-linewidth',
    type=click.FLOAT,
    default=poseconnect.defaults.OVERLAY_KEYPOINT_CONNECTOR_LINEWIDTH,
    help='Width of keypoint connectors in pixels',
    show_default=True
)
@click.option(
    '--pose-label-text-color',
    type=click.STRING,
    default=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_COLOR,
    help='Color for pose label text',
    show_default=True
)
@click.option(
    '--pose-label-box-alpha',
    type=click.FLOAT,
    default=poseconnect.defaults.OVERLAY_POSE_LABEL_BOX_ALPHA,
    help='Transparency of pose label text box',
    show_default=True
)
@click.option(
    '--pose-label-font-scale',
    type=click.FLOAT,
    default=poseconnect.defaults.OVERLAY_POSE_LABEL_FONT_SCALE,
    help='Font scale for pose labels',
    show_default=True
)
@click.option(
    '--pose-label-text-line-width',
    type=click.FLOAT,
    default=poseconnect.defaults.OVERLAY_POSE_LABEL_TEXT_LINE_WIDTH,
    help='Width of lines for pose label text',
    show_default=True
)
@click.option(
    '--draw-timestamp/--no-timestamp',
    default=poseconnect.defaults.OVERLAY_DRAW_TIMESTAMP,
    help='Add a timestamp to each video frame',
    show_default=False
)
@click.option(
    '--timestamp-padding',
    type=click.INT,
    default=poseconnect.defaults.OVERLAY_TIMESTAMP_PADDING,
    help='Separation between timestamp box and upper-right-hand corner of frame in pixels',
    show_default=True
)
@click.option(
    '--timestamp-font-scale',
    type=click.FLOAT,
    default=poseconnect.defaults.OVERLAY_TIMESTAMP_FONT_SCALE,
    help='Font scale for timestamps',
    show_default=True
)
@click.option(
    '--timestamp-text-line-width',
    type=click.FLOAT,
    default=poseconnect.defaults.OVERLAY_TIMESTAMP_TEXT_LINE_WIDTH,
    help='Width of lines for timestamp text',
    show_default=True
)
@click.option(
    '--timestamp-text-color',
    type=click.STRING,
    default=poseconnect.defaults.OVERLAY_TIMESTAMP_TEXT_COLOR,
    help='Color for timestamp text',
    show_default=True
)
@click.option(
    '--timestamp-box-color',
    type=click.STRING,
    default=poseconnect.defaults.OVERLAY_TIMESTAMP_BOX_COLOR,
    help='Color for timestamp text box',
    show_default=True
)
@click.option(
    '--timestamp-box-alpha',
    type=click.FLOAT,
    default=poseconnect.defaults.OVERLAY_TIMESTAMP_BOX_ALPHA,
    help='Transparency of timestamp text box',
    show_default=True
)
@click.option(
    '--progress-bar/--no-progress-bar',
    default=poseconnect.defaults.PROGRESS_BAR,
    required=False,
    help='Display progress bar',
    show_default=False
)
@click.option(
    '--log-level',
    type=click.Choice(
        poseconnect.defaults.LOG_LEVEL_OPTIONS,
        case_sensitive=False
    ),
    default=poseconnect.defaults.LOG_LEVEL,
    help='Log level',
    show_default=True
)
def cli_overlay_poses_video(
    poses_path,
    video_input_path,
    video_start_time,
    video_start_timezone,
    pose_type,
    camera_id,
    camera_calibrations_path,
    pose_label_column,
    pose_label_map_path,
    generate_pose_label_map,
    video_fps,
    video_frame_count,
    video_output_path,
    video_output_directory,
    video_output_filename_suffix,
    video_output_filename_extension,
    video_output_fourcc_string,
    draw_keypoint_connectors,
    pose_model_name,
    pose_color,
    keypoint_radius,
    keypoint_alpha,
    keypoint_connector_alpha,
    keypoint_connector_linewidth,
    pose_label_text_color,
    pose_label_box_alpha,
    pose_label_font_scale,
    pose_label_text_line_width,
    draw_timestamp,
    timestamp_padding,
    timestamp_font_scale,
    timestamp_text_line_width,
    timestamp_text_color,
    timestamp_box_color,
    timestamp_box_alpha,
    progress_bar,
    log_level
):
    if log_level is not None:
        numeric_log_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_log_level, int):
            raise ValueError('Invalid log level: %s'.format(log_level))
        logging.basicConfig(level=numeric_log_level)
    video_start_time = video_start_time.replace(tzinfo=video_start_timezone)
    poseconnect.overlay.overlay_poses_video(
        poses=poses_path,
        video_input_path=video_input_path,
        video_start_time=video_start_time,
        pose_type=pose_type,
        camera_id=camera_id,
        camera_calibration=None,
        camera_calibrations=camera_calibrations_path,
        pose_label_column=pose_label_column,
        pose_label_map=pose_label_map_path,
        generate_pose_label_map=generate_pose_label_map,
        video_fps=video_fps,
        video_frame_count=video_frame_count,
        video_output_path=video_output_path,
        video_output_directory=video_output_directory,
        video_output_filename_suffix=video_output_filename_suffix,
        video_output_filename_extension=video_output_filename_extension,
        video_output_fourcc_string=video_output_fourcc_string,
        draw_keypoint_connectors=draw_keypoint_connectors,
        keypoint_connectors=None,
        pose_model_name=pose_model_name,
        pose_color=pose_color,
        keypoint_radius=keypoint_radius,
        keypoint_alpha=keypoint_alpha,
        keypoint_connector_alpha=keypoint_connector_alpha,
        keypoint_connector_linewidth=keypoint_connector_linewidth,
        pose_label_text_color=pose_label_text_color,
        pose_label_box_alpha=pose_label_box_alpha,
        pose_label_font_scale=pose_label_font_scale,
        pose_label_text_line_width=pose_label_text_line_width,
        draw_timestamp=draw_timestamp,
        timestamp_padding=timestamp_padding,
        timestamp_font_scale=timestamp_font_scale,
        timestamp_text_line_width=timestamp_text_line_width,
        timestamp_text_color=timestamp_text_color,
        timestamp_box_color=timestamp_box_color,
        timestamp_box_alpha=timestamp_box_alpha,
        progress_bar=progress_bar,
        notebook=False
    )

cli.add_command(cli_reconstruct_poses_3d)
cli.add_command(cli_track_poses_3d)
cli.add_command(cli_interpolate_pose_tracks_3d)
cli.add_command(cli_identify_pose_tracks_3d)
cli.add_command(cli_overlay_poses_video)

class TimezoneType(click.ParamType):
    name = "timezone"
    def convert(self, value, param, ctx):
        tzinfo = dateutil.tz.gettz(value)
        if tzinfo is None:
            self.fail('Timezone \'value\' not recognized', param, ctx)
        return tzinfo
