# import process_pose_data.fetch
import smc_kalman
# import cv_utils
# import cv2 as cv
import pandas as pd
import numpy as np
# import networkx as nx
# import tqdm
# import tqdm.notebook
from uuid import uuid4
import logging
import time
import itertools
# from functools import partial

logger = logging.getLogger(__name__)

def generate_pose_tracks(
    poses_3d_df,
    max_match_distance=1.0,
    max_iterations_since_last_match=20,
    centroid_position_initial_sd=1.0,
    centroid_velocity_initial_sd=1.0,
    reference_delta_t_seconds=1.0,
    reference_velocity_drift=1.0,
    position_observation_sd=0.5
):
    poses_3d_df_copy = poses_3d_df.copy()
    poses_3d_df_copy['centroid'] = poses_3d_df_copy['keypoint_coordinates_3d'].apply(
        lambda x: np.nanmean(x, axis=0)
    )
    poses_3d_df_copy['pose_track_3d_id'] = None
    active_tracks = dict()
    inactive_tracks = dict()
    timestamps = np.sort(poses_3d_df['timestamp'].unique())
    for pose_3d_id, row in poses_3d_df_copy.loc[poses_3d_df_copy['timestamp'] == timestamps[0]].iterrows():
        pose_track_3d = PoseTrack3D(
            timestamp=timestamps[0],
            centroid_position=row['centroid'],
            pose_track_2d_ids=row['track_labels'],
            centroid_position_initial_sd=centroid_position_initial_sd,
            centroid_velocity_initial_sd=centroid_velocity_initial_sd,
            reference_delta_t_seconds=reference_delta_t_seconds,
            reference_velocity_drift=reference_velocity_drift,
            position_observation_sd=position_observation_sd
        )
        poses_3d_df_copy.loc[pose_3d_id, 'pose_track_3d_id'] = pose_track_3d.pose_track_3d_id
        active_tracks[pose_track_3d.pose_track_3d_id] = pose_track_3d
    previous_timestamp = timestamps[0]
    for current_timestamp in timestamps[1:]:
        for pose_track_3d in active_tracks.values():
            pose_track_3d.predict(current_timestamp)
        pose_track_3d_ids = list(active_tracks.keys())
        poses_3d_df_copy_timestamp = poses_3d_df_copy.loc[poses_3d_df_copy['timestamp'] == current_timestamp]
        pose_3d_ids = poses_3d_df_copy_timestamp.index.values
        distances_df = pd.DataFrame(
            index = pose_track_3d_ids,
            columns = pose_3d_ids,
            dtype='float'
        )
        for pose_track_3d_id, pose_3d_id in itertools.product(pose_track_3d_ids, pose_3d_ids):
            track_position = active_tracks[pose_track_3d_id].centroid_distribution.mean[:3]
            centroid_position = poses_3d_df_copy.loc[pose_3d_id, 'centroid']
            distance = np.linalg.norm(
                np.subtract(
                    track_position,
                    centroid_position
                )
            )
            if distance < max_match_distance:
                distances_df.loc[pose_track_3d_id, pose_3d_id] = distance
        best_track_for_each_pose = distances_df.idxmin(axis=0)
        best_pose_for_each_track = distances_df.idxmin(axis=1)
        matches = dict(
            set(zip(best_pose_for_each_track.index, best_pose_for_each_track.values)) &
            set(zip(best_track_for_each_pose.values, best_track_for_each_pose.index))
        )
        matched_pose_tracks = set(matches.keys())
        matched_poses = set(matches.values())
        unmatched_pose_tracks = set(active_tracks.keys()) - matched_pose_tracks
        unmatched_poses = set(pose_3d_ids) - matched_poses
        for pose_track_3d_id, pose_3d_id in matches.items():
            poses_3d_df_copy.loc[pose_3d_id, 'pose_track_3d_id'] = pose_track_3d_id
            active_tracks[pose_track_3d_id].iterations_since_last_match = 0
            active_tracks[pose_track_3d_id].incorporate_observation(
                centroid_position=poses_3d_df_copy.loc[pose_3d_id, 'centroid'],
                pose_track_2d_ids=poses_3d_df_copy.loc[pose_3d_id, 'track_labels']
            )
        for unmatched_pose_track_id in unmatched_pose_tracks:
            active_tracks[unmatched_pose_track_id].iterations_since_last_match += 1
            if active_tracks[unmatched_pose_track_id].iterations_since_last_match > max_iterations_since_last_match:
                inactive_tracks[unmatched_pose_track_id] = active_tracks.pop(unmatched_pose_track_id)
        for unmatched_pose_id in unmatched_poses:
            new_pose_track_3d = PoseTrack3D(
                timestamp=current_timestamp,
                centroid_position=poses_3d_df_copy.loc[unmatched_pose_id, 'centroid'],
                pose_track_2d_ids=poses_3d_df_copy.loc[unmatched_pose_id, 'track_labels']
            )
            poses_3d_df_copy.loc[unmatched_pose_id, 'pose_track_3d_id'] = new_pose_track_3d.pose_track_3d_id
            active_tracks[new_pose_track_3d.pose_track_3d_id] = new_pose_track_3d
        previous_timestamp = current_timestamp
    return poses_3d_df_copy, active_tracks, inactive_tracks

class PoseTrack3D:
    def __init__(
        self,
        timestamp,
        centroid_position,
        pose_track_2d_ids,
        centroid_position_initial_sd=1.0,
        centroid_velocity_initial_sd=1.0,
        reference_delta_t_seconds=1.0,
        reference_velocity_drift=1.0,
        position_observation_sd=0.5
    ):
        self.pose_track_3d_id = pose_track_3d_id = uuid4().hex
        self.initial_timestamp = timestamp
        self.latest_timestamp = timestamp
        centroid_position = np.squeeze(np.array(centroid_position))
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
        self.pose_track_2d_ids = set(pose_track_2d_ids)

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
        centroid_position,
        pose_track_2d_ids
    ):

        centroid_position = np.squeeze(np.array(centroid_position))
        self.centroid_distribution = self.centroid_distribution.incorporate_observation(
            linear_gaussian_model=constant_velocity_model(
                delta_t_seconds=None,
                reference_delta_t_seconds=self.reference_delta_t_seconds,
                reference_velocity_drift=self.reference_velocity_drift,
                position_observation_sd=self.position_observation_sd
            ),
            observation_vector=centroid_position
        )
        self.pose_track_2d_ids = self.pose_track_2d_ids.union(pose_track_2d_ids)
        self.centroid_distribution_trajectory['observed_centroid'][-1] = centroid_position
        self.centroid_distribution_trajectory['mean'][-1] = self.centroid_distribution.mean
        self.centroid_distribution_trajectory['covariance'][-1] = self.centroid_distribution.covariance

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
