#!/bin/bash

poseconnect reconstruct \
  sample_pose_2d_data.json \
  sample_camera_calibration_info.json \
  poses_3d.json \
  --pose-model-name COCO-17 \
  --room-x-limits -2.0 7.0 \
  --room-y-limits -2.0 14.0 \
  --progress-bar

poseconnect track \
  poses_3d.json \
  poses_3d_with_tracks.json \
  --progress-bar

poseconnect interpolate \
  poses_3d_with_tracks.json \
  poses_3d_with_tracks_interpolated.json

poseconnect identify \
  poses_3d_with_tracks_interpolated.json \
  sample_sensor_data.json \
  poses_3d_with_tracks_interpolated_identified.json \
  --active-person-id 5c602883-3804-4139-aee1-57ba253266cb \
  --active-person-id 681118d5-5dca-4fbb-80ca-2120b9e6b03b \
  --active-person-id 62a0fd7a-e951-419b-a46a-2dd7b23136c4 \
  --active-person-id 5a4143b0-0541-492b-b703-4540b8096e37 \
  --sensor-position-keypoint-index 5c602883-3804-4139-aee1-57ba253266cb 15 \
  --sensor-position-keypoint-index 681118d5-5dca-4fbb-80ca-2120b9e6b03b 15 \
  --sensor-position-keypoint-index 62a0fd7a-e951-419b-a46a-2dd7b23136c4 15 \
  --sensor-position-keypoint-index 5a4143b0-0541-492b-b703-4540b8096e37 11
