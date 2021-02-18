import smc_kalman
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from uuid import uuid4
import logging
import time
import itertools
import copy

logger = logging.getLogger(__name__)

def update_pose_tracks_3d(
    poses_3d_df,
    pose_tracks_3d=None,
    max_match_distance=1.0,
    max_iterations_since_last_match=20,
    centroid_position_initial_sd=1.0,
    centroid_velocity_initial_sd=1.0,
    reference_delta_t_seconds=1.0,
    reference_velocity_drift=0.30,
    position_observation_sd=0.5,
    progress_bar=False,
    notebook=False
):
    if len(poses_3d_df) == 0:
        return pose_tracks_3d
    if pose_tracks_3d is None:
        initial_timestamp = poses_3d_df['timestamp'].min()
        initial_pose_3d_ids = poses_3d_df.loc[
            poses_3d_df['timestamp'] == initial_timestamp
        ].index.values.tolist()
        initial_keypoint_coordinates_3d = poses_3d_df.loc[
            poses_3d_df['timestamp'] == initial_timestamp,
            'keypoint_coordinates_3d'
        ].values.tolist()
        initial_poses_3d = dict(zip(initial_pose_3d_ids, initial_keypoint_coordinates_3d))
        pose_tracks_3d = PoseTracks3D(
            timestamp=initial_timestamp,
            poses_3d=initial_poses_3d,
            centroid_position_initial_sd=centroid_position_initial_sd,
            centroid_velocity_initial_sd=centroid_velocity_initial_sd,
            reference_delta_t_seconds=reference_delta_t_seconds,
            reference_velocity_drift=reference_velocity_drift,
            position_observation_sd=position_observation_sd
        )
        pose_tracks_3d.update_df(
            poses_3d_df=poses_3d_df.loc[poses_3d_df['timestamp'] != initial_timestamp],
            progress_bar=progress_bar,
            notebook=notebook
        )
    else:
        pose_tracks_3d.update_df(
            poses_3d_df=poses_3d_df,
            progress_bar=progress_bar,
            notebook=notebook
        )
    return pose_tracks_3d

def interpolate_pose_track(pose_track_3d_df):
    if pose_track_3d_df['timestamp'].duplicated().any():
        raise ValueError('Pose data for single pose track contains duplicate timestamps')
    pose_track_3d_df = pose_track_3d_df.copy()
    pose_track_3d_df.sort_values('timestamp', inplace=True)
    old_time_index = pd.DatetimeIndex(pose_track_3d_df['timestamp'])
    combined_time_index = pd.date_range(
        start=pose_track_3d_df['timestamp'].min(),
        end=pose_track_3d_df['timestamp'].max(),
        freq='100ms',
        name='timestamp'
    )
    new_time_index = combined_time_index.difference(old_time_index)
    old_num_poses = len(old_time_index)
    combined_num_poses = len(combined_time_index)
    new_num_poses = len(new_time_index)
    keypoints_flattened_df = pd.DataFrame(
        np.stack(pose_track_3d_df['keypoint_coordinates_3d']).reshape((old_num_poses, -1)),
        index=old_time_index
    )
    keypoints_flattened_interpolated_df = keypoints_flattened_df.reindex(combined_time_index).interpolate(method='time')
    keypoints_flattened_interpolated_array = keypoints_flattened_interpolated_df.values
    keypoints_interpolated_array = keypoints_flattened_interpolated_array.reshape((combined_num_poses, -1, 3))
    keypoints_interpolated_array_unstacked = [keypoints_interpolated_array[i] for i in range(keypoints_interpolated_array.shape[0])]
    poses_3d_interpolated_df = pd.Series(
        keypoints_interpolated_array_unstacked,
        index=combined_time_index,
        name='keypoint_coordinates_3d'
    ).to_frame()
    poses_3d_new_df = poses_3d_interpolated_df.reindex(new_time_index)
    pose_3d_ids_new = [uuid4().hex for _ in range(len(poses_3d_new_df))]
    poses_3d_new_df['pose_3d_id'] = pose_3d_ids_new
    poses_3d_new_df = poses_3d_new_df.reset_index().set_index('pose_3d_id')
    return poses_3d_new_df

class PoseTracks3D:
    def __init__(
        self,
        timestamp,
        poses_3d,
        max_match_distance=1.0,
        max_iterations_since_last_match=20,
        centroid_position_initial_sd=1.0,
        centroid_velocity_initial_sd=1.0,
        reference_delta_t_seconds=1.0,
        reference_velocity_drift=0.30,
        position_observation_sd=0.5
    ):
        self.max_match_distance = max_match_distance
        self.max_iterations_since_last_match = max_iterations_since_last_match
        self.centroid_position_initial_sd = centroid_position_initial_sd
        self.centroid_velocity_initial_sd = centroid_velocity_initial_sd
        self.reference_delta_t_seconds = reference_delta_t_seconds
        self.reference_velocity_drift = reference_velocity_drift
        self.position_observation_sd = position_observation_sd
        self.active_tracks = dict()
        self.inactive_tracks = dict()
        for pose_3d_id, keypoint_coordinates_3d in poses_3d.items():
            pose_track_3d = PoseTrack3D(
                timestamp=timestamp,
                pose_3d_id = pose_3d_id,
                keypoint_coordinates_3d=keypoint_coordinates_3d,
                centroid_position_initial_sd=self.centroid_position_initial_sd,
                centroid_velocity_initial_sd=self.centroid_velocity_initial_sd,
                reference_delta_t_seconds=self.reference_delta_t_seconds,
                reference_velocity_drift=self.reference_velocity_drift,
                position_observation_sd=self.position_observation_sd
            )
            self.active_tracks[pose_track_3d.pose_track_3d_id] = pose_track_3d

    def update_df(
        self,
        poses_3d_df,
        progress_bar=False,
        notebook=False
    ):
        timestamps = np.sort(poses_3d_df['timestamp'].unique())
        if progress_bar:
            if notebook:
                timestamp_iterator = tqdm.notebook.tqdm(timestamps)
            else:
                timestamp_iterator = tqdm.tqdm(timestamps)
        else:
            timestamp_iterator = timestamps
        for current_timestamp in timestamp_iterator:
            current_pose_3d_ids = poses_3d_df.loc[
                poses_3d_df['timestamp'] == current_timestamp
            ].index.values.tolist()
            current_keypoint_coordinates_3d = poses_3d_df.loc[
                poses_3d_df['timestamp'] == current_timestamp,
                'keypoint_coordinates_3d'
            ].values.tolist()
            current_poses_3d = dict(zip(current_pose_3d_ids, current_keypoint_coordinates_3d))
            self.update(
                timestamp=current_timestamp,
                poses_3d=current_poses_3d
            )

    def update(
        self,
        timestamp,
        poses_3d
    ):
        self.predict(
            timestamp=timestamp
        )
        self.incorporate_observations(
            timestamp=timestamp,
            poses_3d=poses_3d
        )

    def predict(
        self,
        timestamp
    ):
        for pose_track_3d in self.active_tracks.values():
            pose_track_3d.predict(timestamp)

    def incorporate_observations(
        self,
        timestamp,
        poses_3d
    ):
        matches = self.match_observations_to_pose_tracks_3d(
            poses_3d=poses_3d
        )
        matched_pose_tracks_3d = set(matches.keys())
        matched_poses = set(matches.values())
        unmatched_pose_tracks_3d = set(self.active_tracks.keys()) - matched_pose_tracks_3d
        unmatched_poses = set(poses_3d.keys()) - matched_poses
        for pose_track_3d_id, pose_3d_id in matches.items():
            self.active_tracks[pose_track_3d_id].iterations_since_last_match = 0
            self.active_tracks[pose_track_3d_id].incorporate_observation(
                pose_3d_id = pose_3d_id,
                keypoint_coordinates_3d = poses_3d[pose_3d_id],
            )
        for pose_track_3d_id in unmatched_pose_tracks_3d:
            self.active_tracks[pose_track_3d_id].iterations_since_last_match += 1
            if self.active_tracks[pose_track_3d_id].iterations_since_last_match > self.max_iterations_since_last_match:
                self.inactive_tracks[pose_track_3d_id] = self.active_tracks.pop(pose_track_3d_id)
        for pose_3d_id in unmatched_poses:
            pose_track_3d = PoseTrack3D(
                timestamp=timestamp,
                pose_3d_id=pose_3d_id,
                keypoint_coordinates_3d=poses_3d[pose_3d_id],
                centroid_position_initial_sd=self.centroid_position_initial_sd,
                centroid_velocity_initial_sd=self.centroid_velocity_initial_sd,
                reference_delta_t_seconds=self.reference_delta_t_seconds,
                reference_velocity_drift=self.reference_velocity_drift,
                position_observation_sd=self.position_observation_sd
            )
            self.active_tracks[pose_track_3d.pose_track_3d_id] = pose_track_3d

    def match_observations_to_pose_tracks_3d(
        self,
        poses_3d
    ):
        pose_track_3d_ids = self.active_tracks.keys()
        pose_3d_ids = poses_3d.keys()
        distances_df = pd.DataFrame(
            index = pose_track_3d_ids,
            columns = pose_3d_ids,
            dtype='float'
        )
        for pose_track_3d_id, pose_3d_id in itertools.product(pose_track_3d_ids, pose_3d_ids):
            track_position = self.active_tracks[pose_track_3d_id].centroid_distribution.mean[:3]
            observation_position = np.nanmean(poses_3d[pose_3d_id], axis=0)
            distance = np.linalg.norm(
                np.subtract(
                    track_position,
                    observation_position
                )
            )
            if distance < self.max_match_distance:
                distances_df.loc[pose_track_3d_id, pose_3d_id] = distance
        best_track_for_each_pose = distances_df.idxmin(axis=0)
        best_pose_for_each_track = distances_df.idxmin(axis=1)
        matches = dict(
            set(zip(best_pose_for_each_track.index, best_pose_for_each_track.values)) &
            set(zip(best_track_for_each_pose.values, best_track_for_each_pose.index))
        )
        return matches

    def filter(
        self,
        num_poses_min=11,
        inplace=False
    ):
        if not inplace:
            new_pose_tracks_3d = copy.deepcopy(self)
        else:
            new_pose_tracks_3d = self
        new_pose_tracks_3d.active_tracks = dict(filter(
            lambda key_value_tuple: key_value_tuple[1].num_poses() >= num_poses_min,
            new_pose_tracks_3d.active_tracks.items()
        ))
        new_pose_tracks_3d.inactive_tracks = dict(filter(
            lambda key_value_tuple: key_value_tuple[1].num_poses() >= num_poses_min,
            new_pose_tracks_3d.inactive_tracks.items()
        ))
        if not inplace:
            return new_pose_tracks_3d

    def extract_pose_tracks_3d(
        self,
        poses_3d_df,
        pose_3d_id_column_name='pose_3d_id',
        pose_track_3d_id_column_name='pose_track_3d_id'
    ):
        input_index_name = poses_3d_df.index.name
        poses_3d_with_tracks_df = poses_3d_df.join(
            self.output_df(
                pose_3d_id_column_name=pose_3d_id_column_name,
                pose_track_3d_id_column_name=pose_track_3d_id_column_name
            ),
            how='inner'
        )
        poses_3d_with_tracks_df.index.name = input_index_name
        return poses_3d_with_tracks_df

    def output(self):
        output = {pose_track_3d_id: pose_track_3d.output() for pose_track_3d_id, pose_track_3d in self.tracks().items()}
        return output

    def output_df(
        self,
        pose_3d_id_column_name='pose_3d_id',
        pose_track_3d_id_column_name='pose_track_3d_id'
    ):
        df = pd.concat(
            [pose_track_3d.output_df(
                pose_3d_id_column_name=pose_3d_id_column_name,
                pose_track_3d_id_column_name=pose_track_3d_id_column_name
            ) for pose_track_3d in self.tracks().values()]
        )
        return df

    def tracks(self):
        return {**self.active_tracks, **self.inactive_tracks}

    def plot_trajectories(
        self,
        pose_track_3d_ids,
        track_label_lookup=None,
        fig_width_inches=8.0,
        fig_height_inches=10.5,
        show=True
    ):
        if track_label_lookup is None:
            track_label_lookup = {pose_track_3d_id: pose_track_3d_id[:2] for pose_track_3d_id in pose_track_3d_ids}
        fig, axes = plt.subplots(3, 1, sharex=True)
        for pose_track_3d_id in pose_track_3d_ids:
            for axis_index, axis_name in enumerate(['x', 'y', 'z']):
                self.tracks()[pose_track_3d_id].draw_trajectory(
                    axis_index=axis_index,
                    axis_name=axis_name,
                    axis_object=axes[axis_index],
                    track_label_lookup=track_label_lookup
                )
        axes[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        axes[2].set_xlabel('Time')
        fig.autofmt_xdate()
        fig.set_size_inches(fig_width_inches, fig_height_inches)
        if show:
            plt.show()

class PoseTrack3D:
    def __init__(
        self,
        timestamp,
        pose_3d_id,
        keypoint_coordinates_3d,
        centroid_position_initial_sd=1.0,
        centroid_velocity_initial_sd=1.0,
        reference_delta_t_seconds=1.0,
        reference_velocity_drift=0.30,
        position_observation_sd=0.5
    ):
        keypoint_coordinates_3d = np.asarray(keypoint_coordinates_3d)
        if keypoint_coordinates_3d.ndim != 2:
            raise ValueError('Keypoint coordinate array should be two dimensional (Number of keypoints x 3)')
        centroid_position = np.nanmean(keypoint_coordinates_3d, axis=0)
        self.pose_track_3d_id = pose_track_3d_id = uuid4().hex
        self.initial_timestamp = timestamp
        self.latest_timestamp = timestamp
        self.pose_3d_ids = [pose_3d_id]
        self.centroid_distribution = smc_kalman.GaussianDistribution(
            mean=np.concatenate((centroid_position.reshape((3,)), np.repeat(0.0, 3))),
            covariance=np.diag(np.concatenate((
                np.repeat(centroid_position_initial_sd**2, 3),
                np.repeat(centroid_velocity_initial_sd**2, 3)
            )))
        )
        self.reference_delta_t_seconds = reference_delta_t_seconds
        self.reference_velocity_drift = reference_velocity_drift
        self.position_observation_sd = position_observation_sd
        self.iterations_since_last_match = 0
        self.centroid_distribution_trajectory = {
            'timestamp': [self.latest_timestamp],
            'observed_centroid': [centroid_position],
            'mean': [self.centroid_distribution.mean],
            'covariance': [self.centroid_distribution.covariance]
        }

    def predict(
        self,
        timestamp
    ):
        delta_t_seconds = (timestamp - self.latest_timestamp).total_seconds()
        self.centroid_distribution = self.centroid_distribution.predict(
            linear_gaussian_model=constant_velocity_model(
                delta_t_seconds=delta_t_seconds,
                reference_delta_t_seconds=self.reference_delta_t_seconds,
                reference_velocity_drift=self.reference_velocity_drift,
                position_observation_sd=self.position_observation_sd
            )
        )
        self.latest_timestamp=timestamp
        self.centroid_distribution_trajectory['timestamp'].append(self.latest_timestamp)
        self.centroid_distribution_trajectory['observed_centroid'].append(np.array([np.nan, np.nan, np.nan]))
        self.centroid_distribution_trajectory['mean'].append(self.centroid_distribution.mean)
        self.centroid_distribution_trajectory['covariance'].append(self.centroid_distribution.covariance)

    def incorporate_observation(
        self,
        pose_3d_id,
        keypoint_coordinates_3d
    ):
        keypoint_coordinates_3d = np.asarray(keypoint_coordinates_3d)
        if keypoint_coordinates_3d.ndim != 2:
            raise ValueError('Keypoint coordinate array should be two dimensional (Number of keypoints x 3)')
        centroid_position = np.nanmean(keypoint_coordinates_3d, axis=0)
        self.pose_3d_ids.append(pose_3d_id)
        self.centroid_distribution = self.centroid_distribution.incorporate_observation(
            linear_gaussian_model=constant_velocity_model(
                delta_t_seconds=None,
                reference_delta_t_seconds=self.reference_delta_t_seconds,
                reference_velocity_drift=self.reference_velocity_drift,
                position_observation_sd=self.position_observation_sd
            ),
            observation_vector=centroid_position
        )
        self.centroid_distribution_trajectory['observed_centroid'][-1] = centroid_position
        self.centroid_distribution_trajectory['mean'][-1] = self.centroid_distribution.mean
        self.centroid_distribution_trajectory['covariance'][-1] = self.centroid_distribution.covariance

    def num_poses(
        self
    ):
        return(len(self.pose_3d_ids))

    def centroid_distribution_trajectory_df(self):
        df = pd.DataFrame({
            'timestamp': self.centroid_distribution_trajectory['timestamp'],
            'observed_centroid': self.centroid_distribution_trajectory['observed_centroid'],
            'position': [mean[:3] for mean in self.centroid_distribution_trajectory['mean']],
            'velocity': [mean[3:] for mean in self.centroid_distribution_trajectory['mean']],
            'covariance': self.centroid_distribution_trajectory['covariance']
        })
        df.set_index('timestamp', inplace=True)
        return df

    def output(self):
        output = {
            'start': pd.to_datetime(self.initial_timestamp).to_pydatetime(),
            'end': pd.to_datetime(self.latest_timestamp).to_pydatetime(),
            'pose_3d_ids': self.pose_3d_ids
        }
        return output

    def output_df(
        self,
        pose_3d_id_column_name='pose_3d_id',
        pose_track_3d_id_column_name='pose_track_3d_id'
    ):
        df = pd.DataFrame([
            {pose_3d_id_column_name: pose_id, pose_track_3d_id_column_name: self.pose_track_3d_id}
            for pose_id in self.pose_3d_ids
        ]).set_index(pose_3d_id_column_name)
        return df

    def plot_trajectory(
        self,
        track_label_lookup=None,
        fig_width_inches=8.0,
        fig_height_inches=10.5,
        show=True
    ):
        if track_label_lookup is None:
            track_label_lookup = {self.pose_track_3d_id: self.pose_track_3d_id[:2]}
        fig, axes = plt.subplots(3, 1, sharex=True)
        for axis_index, axis_name in enumerate(['x', 'y', 'z']):
            self.draw_trajectory(
                axis_index=axis_index,
                axis_name=axis_name,
                axis_object=axes[axis_index],
                track_label_lookup=track_label_lookup
            )
        axes[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        axes[2].set_xlabel('Time')
        fig.autofmt_xdate()
        fig.set_size_inches(fig_width_inches, fig_height_inches)
        if show:
            plt.show()

    def draw_trajectory(
        self,
        axis_index,
        axis_name,
        axis_object,
        track_label_lookup=None
    ):
        if track_label_lookup is None:
            track_label_lookup = {self.pose_track_3d_id: self.pose_track_3d_id[:2]}
        df = self.centroid_distribution_trajectory_df()
        axis_object.fill_between(
            df.index,
            np.stack(df['position'])[:, axis_index] - np.sqrt(np.stack(df['covariance'])[:, axis_index, axis_index]),
            np.stack(df['position'])[:, axis_index] + np.sqrt(np.stack(df['covariance'])[:, axis_index, axis_index]),
            alpha = 0.4,
            label='Track {} confidence interval'.format(track_label_lookup[self.pose_track_3d_id])
        )
        axis_object.plot(
            df.index,
            np.stack(df['observed_centroid'])[:, axis_index],
            '.',
            label='Track {} observation'.format(track_label_lookup[self.pose_track_3d_id])
        )
        axis_object.set_ylabel('${}$ position (meters)'.format(axis_name))

def constant_velocity_model(
    delta_t_seconds,
    reference_delta_t_seconds=1.0,
    reference_velocity_drift=1.0,
    position_observation_sd=0.5
):
    if delta_t_seconds is not None:
        velocity_drift = reference_velocity_drift*np.sqrt(delta_t_seconds/reference_delta_t_seconds)
    else:
        delta_t_seconds = 0.0
        velocity_drift = 0.0
    model = smc_kalman.LinearGaussianModel(
        transition_model = np.array([
            [1.0, 0.0, 0.0, delta_t_seconds, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, delta_t_seconds, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, delta_t_seconds],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]),
        transition_noise_covariance = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, velocity_drift**2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, velocity_drift**2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, velocity_drift**2]
        ]),
        observation_model = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ]),
        observation_noise_covariance = np.array([
            [position_observation_sd**2, 0.0, 0.0],
            [0.0, position_observation_sd**2, 0.0],
            [0.0, 0.0, position_observation_sd**2]
        ]),
        control_model = None
    )
    return model
