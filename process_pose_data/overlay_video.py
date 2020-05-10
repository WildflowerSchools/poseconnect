import process_pose_data.visualize
import video_io
import numpy as np
import cv_utils
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import string

def draw_poses_2d_timestamp_camera_pair_opencv(
    df,
    annotate_matches=False,
    generate_match_aliases=False,
    camera_names={'a': None, 'b': None},
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    keypoint_alpha=0.3,
    keypoint_connector_alpha=0.3,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.5,
    pose_label_font_scale=2.0,
    show=True,
    fig_width_inches=10.5,
    fig_height_inches=8
):
    dfs_single_camera = process_pose_data.visualize.extract_single_camera_data(df)
    pose_label_maps = {
        'a': None,
        'b': None
    }
    pose_color_maps = {
        'a': None,
        'b': None
    }
    if annotate_matches:
        num_matches = df['match'].sum()
        for camera_letter in ['a', 'b']:
            pose_ids = dfs_single_camera[camera_letter].index.values.tolist()
            pose_label_maps[camera_letter] = {pose_id: '' for pose_id in pose_ids}
            if not generate_match_aliases:
                if 'track_label' in dfs_single_camera[camera_letter].columns:
                    pose_labels = dfs_single_camera[camera_letter]['track_label'].values.tolist()
                elif len(pose_ids) <= 26:
                    pose_labels = string.ascii_uppercase[:len(pose_ids)]
                else:
                    pose_labels = range(len(pose_ids))
                pose_label_maps[camera_letter] = dict(zip(pose_ids, pose_labels))
            pose_color_maps[camera_letter] = {pose_id: 'grey' for pose_id in pose_ids}
        match_aliases = iter(list(string.ascii_uppercase[:num_matches]))
        match_colors = iter(sns.color_palette('husl', n_colors=num_matches))
        for (pose_id_a, pose_id_b), row in df.iterrows():
            if row['match']:
                old_label_a = pose_label_maps['a'][pose_id_a]
                old_label_b = pose_label_maps['b'][pose_id_b]
                pose_label_maps['a'][pose_id_a] = '{} ({})'.format(
                    old_label_a,
                    old_label_b
                )
                pose_label_maps['b'][pose_id_b] = '{} ({})'.format(
                    old_label_b,
                    old_label_a
                )
                if generate_match_aliases:
                    match_alias = next(match_aliases)
                    pose_label_maps['a'][pose_id_a] = match_alias
                    pose_label_maps['b'][pose_id_b] = match_alias
                pose_color = next(match_colors)
                pose_color_maps['a'][pose_id_a] = pose_color
                pose_color_maps['b'][pose_id_b] = pose_color
    for camera_letter in ['a', 'b']:
        draw_poses_2d_timestamp_camera_opencv(
            df=dfs_single_camera[camera_letter],
            draw_keypoint_connectors=draw_keypoint_connectors,
            keypoint_connectors=keypoint_connectors,
            pose_label_map=pose_label_maps[camera_letter],
            pose_color_map=pose_color_maps[camera_letter],
            keypoint_alpha=keypoint_alpha,
            keypoint_connector_alpha=keypoint_connector_alpha,
            keypoint_connector_linewidth=keypoint_connector_linewidth,
            pose_label_color=pose_label_color,
            pose_label_background_alpha=pose_label_background_alpha,
            pose_label_font_scale=pose_label_font_scale,
            show=show,
            fig_width_inches=fig_width_inches,
            fig_height_inches=fig_height_inches
        )

def draw_poses_2d_timestamp_camera_opencv(
    df,
    background_image=None,
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    pose_label_map=None,
    pose_color_map=None,
    keypoint_alpha=0.3,
    keypoint_connector_alpha=0.3,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.5,
    pose_label_font_scale=2.0,
    show=True,
    fig_width_inches=10.5,
    fig_height_inches=8
):
    timestamps = df['timestamp'].unique()
    camera_ids = df['camera_id'].unique()
    if len(timestamps) > 1:
        raise ValueError('More than one timestamp in data frame')
    if len(camera_ids) > 1:
        raise ValueError('More than one camera in data frame')
    timestamp = timestamps[0]
    camera_id = camera_ids[0]
    camera_identifier = camera_id
    df = df.sort_index()
    pose_ids = df.index.tolist()
    if pose_label_map is None:
        if 'track_label' in df.columns:
            pose_labels = df['track_label'].values.tolist()
        elif len(pose_ids) <= 26:
            pose_labels = string.ascii_uppercase[:len(pose_ids)]
        else:
            pose_labels = range(len(pose_ids))
        pose_label_map = dict(zip(pose_ids, pose_labels))
    if pose_color_map is None:
        pose_colors = sns.color_palette('husl', n_colors=len(pose_ids))
        pose_color_map = dict(zip(pose_ids, pose_colors))
    if draw_keypoint_connectors:
        if keypoint_connectors is None:
            pose_model = process_pose_data.fetch.fetch_pose_model(
                pose_id=pose_ids[0]
            )
            keypoint_connectors = pose_model.get('keypoint_connectors')
    if background_image is None:
        image_metadata = video_io.fetch_images(
            image_timestamps = [timestamp.to_pydatetime()],
            camera_device_ids=[camera_id]
        )
        image_local_path = image_metadata[0]['image_local_path']
        background_image = cv_utils.fetch_image_from_local_drive(image_local_path)
    new_image = background_image
    for pose_id, row in df.iterrows():
        new_image = process_pose_data.draw_pose_2d_opencv(
            image=new_image,
            keypoint_coordinates=row['keypoint_coordinates'],
            draw_keypoint_connectors=draw_keypoint_connectors,
            keypoint_connectors=keypoint_connectors,
            pose_label=pose_label_map[pose_id],
            pose_color=pose_color_map[pose_id],
            keypoint_alpha=keypoint_alpha,
            keypoint_connector_alpha=keypoint_connector_alpha,
            keypoint_connector_linewidth=keypoint_connector_linewidth,
            pose_label_color=pose_label_color,
            pose_label_background_alpha=pose_label_background_alpha,
            pose_label_font_scale=pose_label_font_scale
        )
    # Show plot
    if show:
        fig, ax = plt.subplots()
        ax.imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))
        ax.axis('off')
        fig.set_size_inches(fig_width_inches, fig_height_inches)
    return new_image

def draw_pose_2d_opencv(
    image,
    keypoint_coordinates,
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    pose_label=None,
    pose_color='green',
    keypoint_radius=3,
    keypoint_alpha=0.3,
    keypoint_connector_alpha=0.3,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.5,
    pose_label_font_scale=2.0,
    pose_label_line_width=2
):
    pose_color = matplotlib.colors.to_hex(pose_color, keep_alpha=False)
    pose_label_color = matplotlib.colors.to_hex(pose_label_color, keep_alpha=False)
    keypoint_coordinates = np.asarray(keypoint_coordinates).reshape((-1, 2))
    valid_keypoints = np.all(np.isfinite(keypoint_coordinates), axis=1)
    plottable_points = keypoint_coordinates[valid_keypoints]
    new_image = image
    for point_index in range(plottable_points.shape[0]):
        new_image = cv_utils.draw_circle(
            original_image=new_image,
            coordinates=plottable_points[point_index],
            radius=keypoint_radius,
            line_width=1,
            color=pose_color,
            fill=True,
            alpha=keypoint_alpha
        )
    if draw_keypoint_connectors and (keypoint_connectors is not None):
        for keypoint_connector in keypoint_connectors:
            keypoint_from_index = keypoint_connector[0]
            keypoint_to_index = keypoint_connector[1]
            if valid_keypoints[keypoint_from_index] and valid_keypoints[keypoint_to_index]:
                new_image=cv_utils.draw_line(
                    original_image=new_image,
                    coordinates=[
                        keypoint_coordinates[keypoint_from_index],
                        keypoint_coordinates[keypoint_to_index]
                    ],
                    line_width=keypoint_connector_linewidth,
                    color=pose_color,
                    alpha=keypoint_connector_alpha
                )
    if pose_label is not None:
        pose_label_anchor = np.nanmean(keypoint_coordinates, axis=0)
        text_box_size, baseline = cv.getTextSize(
            text=str(pose_label),
            fontFace=cv.FONT_HERSHEY_PLAIN,
            fontScale=pose_label_font_scale,
            thickness=pose_label_line_width
        )
        new_image=cv_utils.draw_rectangle(
            original_image=new_image,
            coordinates=[
                [
                    pose_label_anchor[0] - text_box_size[0]/2,
                    pose_label_anchor[1] - (text_box_size[1] + baseline)/2
                ],
                [
                    pose_label_anchor[0] + text_box_size[0]/2,
                    pose_label_anchor[1] + (text_box_size[1] + baseline)/2
                ]
            ],
            line_width=1.5,
            color=pose_color,
            fill=True,
            alpha=pose_label_background_alpha
        )
        new_image=cv_utils.draw_text(
            original_image=new_image,
            coordinates=pose_label_anchor,
            text=str(pose_label),
            horizontal_alignment='center',
            vertical_alignment='middle',
            font_face=cv.FONT_HERSHEY_PLAIN,
            font_scale=pose_label_font_scale,
            line_width=pose_label_line_width,
            color=pose_label_color
        )
    return new_image
