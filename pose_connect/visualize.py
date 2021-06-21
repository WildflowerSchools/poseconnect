import honeycomb_io
import cv_utils
import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import slugify
import string
import os
import logging

logger = logging.getLogger(__name__)

register_matplotlib_converters()

def keypoint_quality_histogram_by_camera(
    df,
    display_camera_name=False,
    camera_names=None,
    bins=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='keypoint_quality',
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if display_camera_name:
        if camera_names is None:
            camera_ids = df['camera_id'].unique().tolist()
            camera_names = honeycomb_io.fetch_camera_names(camera_ids)
    for camera_id, group_df in df.groupby('camera_id'):
        if display_camera_name:
            camera_id_string = camera_names.get(camera_id)
        else:
            camera_id_string = camera_id
        plot_title = camera_id_string
        file_identifier = camera_id_string
        keypoint_quality_histogram(
            df=group_df,
            bins=bins,
            plot_title=plot_title,
            plot_title_datetime_format=plot_title_datetime_format,
            show=show,
            save=save,
            save_directory=save_directory,
            filename_prefix=filename_prefix,
            file_identifier=file_identifier,
            filename_datetime_format=filename_datetime_format,
            filename_extension=filename_extension,
            fig_width_inches=fig_width_inches,
            fig_height_inches=fig_height_inches
        )

def keypoint_quality_histogram(
    df,
    bins=None,
    plot_title=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='keypoint_quality',
    file_identifier=None,
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if plot_title is not None:
        fig_suptitle = '{} ({} - {})'.format(
            plot_title,
            df['timestamp'].min().strftime(plot_title_datetime_format),
            df['timestamp'].max().strftime(plot_title_datetime_format)
        )
    else:
        fig_suptitle = '{} - {}'.format(
            df['timestamp'].min().strftime(plot_title_datetime_format),
            df['timestamp'].max().strftime(plot_title_datetime_format)
        )
    if file_identifier is not None:
        save_filename = '{}_{}_{}_{}.{}'.format(
            filename_prefix,
            slugify.slugify(file_identifier),
            df['timestamp'].min().strftime(filename_datetime_format),
            df['timestamp'].max().strftime(filename_datetime_format),
            filename_extension
        )
    else:
        save_filename = '{}_{}_{}.{}'.format(
            filename_prefix,
            df['timestamp'].min().strftime(filename_datetime_format),
            df['timestamp'].max().strftime(filename_datetime_format),
            filename_extension
        )
    sns_plot = sns.distplot(
        np.concatenate(df['keypoint_quality_2d'].values),
        bins=bins,
        kde=False
    )
    plt.xlabel('Keypoint quality')
    plt.ylabel('Number of keypoints')
    fig = sns_plot.get_figure()
    fig.suptitle(fig_suptitle)
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    # Show plot
    if show:
        plt.show()
    # Save plot
    if save:
        path = os.path.join(
            save_directory,
            save_filename
        )
        fig.savefig(path)

def num_valid_keypoints_histogram_by_camera(
    df,
    display_camera_name=False,
    camera_names=None,
    bins=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='num_valid_keypoints',
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if display_camera_name:
        if camera_names is None:
            camera_ids = df['camera_id'].unique().tolist()
            camera_names = honeycomb_io.fetch_camera_names(camera_ids)
    for camera_id, group_df in df.groupby('camera_id'):
        if display_camera_name:
            camera_id_string = camera_names.get(camera_id)
        else:
            camera_id_string = camera_id
        plot_title = camera_id_string
        file_identifier = camera_id_string
        num_valid_keypoints_histogram(
            df=group_df,
            bins=bins,
            plot_title=plot_title,
            plot_title_datetime_format=plot_title_datetime_format,
            show=show,
            save=save,
            save_directory=save_directory,
            filename_prefix=filename_prefix,
            file_identifier=file_identifier,
            filename_datetime_format=filename_datetime_format,
            filename_extension=filename_extension,
            fig_width_inches=fig_width_inches,
            fig_height_inches=fig_height_inches
        )

def num_valid_keypoints_histogram(
    df,
    bins=None,
    plot_title=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='num_valid_keypoints',
    file_identifier=None,
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if plot_title is not None:
        fig_suptitle = '{} ({} - {})'.format(
            plot_title,
            df['timestamp'].min().strftime(plot_title_datetime_format),
            df['timestamp'].max().strftime(plot_title_datetime_format)
        )
    else:
        fig_suptitle = '{} - {}'.format(
            df['timestamp'].min().strftime(plot_title_datetime_format),
            df['timestamp'].max().strftime(plot_title_datetime_format)
        )
    if file_identifier is not None:
        save_filename = '{}_{}_{}_{}.{}'.format(
            filename_prefix,
            slugify.slugify(file_identifier),
            df['timestamp'].min().strftime(filename_datetime_format),
            df['timestamp'].max().strftime(filename_datetime_format),
            filename_extension
        )
    else:
        save_filename = '{}_{}_{}.{}'.format(
            filename_prefix,
            df['timestamp'].min().strftime(filename_datetime_format),
            df['timestamp'].max().strftime(filename_datetime_format),
            filename_extension
        )
    num_valid_keypoints_series = df['keypoint_quality_2d'].apply(lambda x: np.count_nonzero(~np.isnan(x)))
    max_num_valid_keypoints = num_valid_keypoints_series.max()
    sns_plot = sns.countplot(
        x = num_valid_keypoints_series,
        order = range(max_num_valid_keypoints + 1),
        color=sns.color_palette()[0]
    )
    plt.xlabel('Number of valid keypoints')
    plt.ylabel('Number of poses')
    fig = sns_plot.get_figure()
    fig.suptitle(fig_suptitle)
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    # Show plot
    if show:
        plt.show()
    # Save plot
    if save:
        path = os.path.join(
            save_directory,
            save_filename
        )
        fig.savefig(path)

def pose_quality_histogram_by_camera(
    df,
    display_camera_name=False,
    camera_names=None,
    bins=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='pose_quality',
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if display_camera_name:
        if camera_names is None:
            camera_ids = df['camera_id'].unique().tolist()
            camera_names = honeycomb_io.fetch_camera_names(camera_ids)
    for camera_id, group_df in df.groupby('camera_id'):
        if display_camera_name:
            camera_id_string = camera_names.get(camera_id)
        else:
            camera_id_string = camera_id
        plot_title = camera_id_string
        file_identifier = camera_id_string
        pose_quality_histogram(
            df=group_df,
            bins=bins,
            plot_title=plot_title,
            plot_title_datetime_format=plot_title_datetime_format,
            show=show,
            save=save,
            save_directory=save_directory,
            filename_prefix=filename_prefix,
            file_identifier=file_identifier,
            filename_datetime_format=filename_datetime_format,
            filename_extension=filename_extension,
            fig_width_inches=fig_width_inches,
            fig_height_inches=fig_height_inches
        )

def pose_quality_histogram(
    df,
    bins=None,
    plot_title=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='pose_quality',
    file_identifier=None,
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if plot_title is not None:
        fig_suptitle = '{} ({} - {})'.format(
            plot_title,
            df['timestamp'].min().strftime(plot_title_datetime_format),
            df['timestamp'].max().strftime(plot_title_datetime_format)
        )
    else:
        fig_suptitle = '{} - {}'.format(
            df['timestamp'].min().strftime(plot_title_datetime_format),
            df['timestamp'].max().strftime(plot_title_datetime_format)
        )
    if file_identifier is not None:
        save_filename = '{}_{}_{}_{}.{}'.format(
            filename_prefix,
            slugify.slugify(file_identifier),
            df['timestamp'].min().strftime(filename_datetime_format),
            df['timestamp'].max().strftime(filename_datetime_format),
            filename_extension
        )
    else:
        save_filename = '{}_{}_{}.{}'.format(
            filename_prefix,
            df['timestamp'].min().strftime(filename_datetime_format),
            df['timestamp'].max().strftime(filename_datetime_format),
            filename_extension
        )
    sns_plot = sns.distplot(
        df['pose_quality_2d'],
        bins=bins,
        kde=False
    )
    plt.xlabel('Pose quality')
    plt.ylabel('Number of poses')
    fig = sns_plot.get_figure()
    fig.suptitle(fig_suptitle)
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    # Show plot
    if show:
        plt.show()
    # Save plot
    if save:
        path = os.path.join(
            save_directory,
            save_filename
        )
        fig.savefig(path)

def mean_keypoint_quality_histogram_by_camera(
    df,
    display_camera_name=False,
    camera_names=None,
    bins=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='mean_keypoint_quality',
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if display_camera_name:
        if camera_names is None:
            camera_ids = df['camera_id'].unique().tolist()
            camera_names = honeycomb_io.fetch_camera_names(camera_ids)
    for camera_id, group_df in df.groupby('camera_id'):
        if display_camera_name:
            camera_id_string = camera_names.get(camera_id)
        else:
            camera_id_string = camera_id
        plot_title = camera_id_string
        file_identifier = camera_id_string
        mean_keypoint_quality_histogram(
            df=group_df,
            bins=bins,
            plot_title=plot_title,
            plot_title_datetime_format=plot_title_datetime_format,
            show=show,
            save=save,
            save_directory=save_directory,
            filename_prefix=filename_prefix,
            file_identifier=file_identifier,
            filename_datetime_format=filename_datetime_format,
            filename_extension=filename_extension,
            fig_width_inches=fig_width_inches,
            fig_height_inches=fig_height_inches
        )

def mean_keypoint_quality_histogram(
    df,
    bins=None,
    plot_title=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='mean_keypoint_quality',
    file_identifier=None,
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if plot_title is not None:
        fig_suptitle = '{} ({} - {})'.format(
            plot_title,
            df['timestamp'].min().strftime(plot_title_datetime_format),
            df['timestamp'].max().strftime(plot_title_datetime_format)
        )
    else:
        fig_suptitle = '{} - {}'.format(
            df['timestamp'].min().strftime(plot_title_datetime_format),
            df['timestamp'].max().strftime(plot_title_datetime_format)
        )
    if file_identifier is not None:
        save_filename = '{}_{}_{}_{}.{}'.format(
            filename_prefix,
            slugify.slugify(file_identifier),
            df['timestamp'].min().strftime(filename_datetime_format),
            df['timestamp'].max().strftime(filename_datetime_format),
            filename_extension
        )
    else:
        save_filename = '{}_{}_{}.{}'.format(
            filename_prefix,
            df['timestamp'].min().strftime(filename_datetime_format),
            df['timestamp'].max().strftime(filename_datetime_format),
            filename_extension
        )
    sns_plot = sns.distplot(
        df['keypoint_quality_2d'].apply(lambda x: np.nanmean(x)),
        bins=bins,
        kde=False
    )
    plt.xlabel('Mean keypoint quality')
    plt.ylabel('Number of poses')
    fig = sns_plot.get_figure()
    fig.suptitle(fig_suptitle)
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    # Show plot
    if show:
        plt.show()
    # Save plot
    if save:
        path = os.path.join(
            save_directory,
            save_filename
        )
        fig.savefig(path)

def mean_keypoint_quality_pose_quality_scatter_by_camera(
    df,
    display_camera_name=False,
    camera_names=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='mean_keypoint_quality_pose_quality',
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if display_camera_name:
        if camera_names is None:
            camera_ids = df['camera_id'].unique().tolist()
            camera_names = honeycomb_io.fetch_camera_names(camera_ids)
    for camera_id, group_df in df.groupby('camera_id'):
        if display_camera_name:
            camera_id_string = camera_names.get(camera_id)
        else:
            camera_id_string = camera_id
        plot_title = camera_id_string
        file_identifier = camera_id_string
        mean_keypoint_quality_pose_quality_scatter(
            df=group_df,
            plot_title=plot_title,
            plot_title_datetime_format=plot_title_datetime_format,
            show=show,
            save=save,
            save_directory=save_directory,
            filename_prefix=filename_prefix,
            file_identifier=file_identifier,
            filename_datetime_format=filename_datetime_format,
            filename_extension=filename_extension,
            fig_width_inches=fig_width_inches,
            fig_height_inches=fig_height_inches
        )

def mean_keypoint_quality_pose_quality_scatter(
    df,
    plot_title=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='mean_keypoint_quality_pose_quality_scatter',
    file_identifier=None,
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if plot_title is not None:
        fig_suptitle = '{} ({} - {})'.format(
            plot_title,
            df['timestamp'].min().strftime(plot_title_datetime_format),
            df['timestamp'].max().strftime(plot_title_datetime_format)
        )
    else:
        fig_suptitle = '{} - {}'.format(
            df['timestamp'].min().strftime(plot_title_datetime_format),
            df['timestamp'].max().strftime(plot_title_datetime_format)
        )
    if file_identifier is not None:
        save_filename = '{}_{}_{}_{}.{}'.format(
            filename_prefix,
            slugify.slugify(file_identifier),
            df['timestamp'].min().strftime(filename_datetime_format),
            df['timestamp'].max().strftime(filename_datetime_format),
            filename_extension
        )
    else:
        save_filename = '{}_{}_{}.{}'.format(
            filename_prefix,
            df['timestamp'].min().strftime(filename_datetime_format),
            df['timestamp'].max().strftime(filename_datetime_format),
            filename_extension
        )
    sns_plot = sns.scatterplot(
        x=df['keypoint_quality_2d'].apply(lambda x: np.nanmean(x)),
        y=df['pose_quality_2d']
    )
    plt.xlabel('Mean keypoint quality')
    plt.ylabel('Pose quality')
    fig = sns_plot.get_figure()
    fig.suptitle(fig_suptitle)
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    # Show plot
    if show:
        plt.show()
    # Save plot
    if save:
        path = os.path.join(
            save_directory,
            save_filename
        )
        fig.savefig(path)

def pose_pair_score_histogram(
    df,
    bins=None,
    plot_title=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='pose_pair_score',
    file_identifier=None,
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    timestamp_min = df['timestamp'].min()
    timestamp_max = df['timestamp'].max()
    if plot_title is not None:
        fig_suptitle = '{} ({} - {})'.format(
            plot_title,
            timestamp_min.strftime(plot_title_datetime_format),
            timestamp_max.strftime(plot_title_datetime_format)
        )
    else:
        fig_suptitle = '{} - {}'.format(
            timestamp_min.strftime(plot_title_datetime_format),
            timestamp_max.strftime(plot_title_datetime_format)
        )
    if file_identifier is not None:
        save_filename = '{}_{}_{}_{}.{}'.format(
            filename_prefix,
            slugify.slugify(file_identifier),
            timestamp_min.strftime(filename_datetime_format),
            timestamp_max.strftime(filename_datetime_format),
            filename_extension
        )
    else:
        save_filename = '{}_{}_{}.{}'.format(
            filename_prefix,
            timestamp_min.strftime(filename_datetime_format),
            timestamp_max.strftime(filename_datetime_format),
            filename_extension
        )
    sns_plot = sns.distplot(
        df['score'],
        bins=bins,
        kde=False
    )
    plt.xlabel('Pose pair score')
    plt.ylabel('Number of pose pairs')
    fig = sns_plot.get_figure()
    fig.suptitle(fig_suptitle)
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    # Show plot
    if show:
        plt.show()
    # Save plot
    if save:
        path = os.path.join(
            save_directory,
            save_filename
        )
        fig.savefig(path)

def pose_pair_score_heatmap_timestamp_camera_pair(
    df,
    min_score=None,
    max_score=None,
    color_rule='score',
    score_color_map_name='summer_r',
    pose_label_map_a=None,
    pose_label_map_b=None,
    display_camera_names=False,
    camera_names=None,
    plot_title=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S.%f',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='pose_pair_score_heatmap',
    file_identifier=None,
    filename_datetime_format='%Y%m%d_%H%M%S_%f',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    timestamps = df['timestamp'].unique()
    camera_ids_a = df['camera_id_a'].unique()
    camera_ids_b = df['camera_id_b'].unique()
    if len(timestamps) > 1:
        raise ValueError('More than one timestamp in data frame')
    if len(camera_ids_a) > 1:
        raise ValueError('More than one camera A in data frame')
    if len(camera_ids_b) > 1:
        raise ValueError('More than one camera B in data frame')
    timestamp = timestamps[0]
    camera_id_a = camera_ids_a[0]
    camera_id_b = camera_ids_b[0]
    if display_camera_names:
        if camera_names is None:
            camera_ids = [camera_id_a, camera_id_b]
            camera_names = honeycomb_io.fetch_camera_names(camera_ids)
    if plot_title is not None:
        ax_title = '{} ({})'.format(
            plot_title,
            timestamp.strftime(plot_title_datetime_format)
        )
    else:
        ax_title = '{}'.format(
            timestamp.strftime(plot_title_datetime_format)
        )
    if file_identifier is not None:
        save_filename = '{}_{}_{}.{}'.format(
            filename_prefix,
            slugify.slugify(file_identifier),
            timestamp.strftime(filename_datetime_format),
            filename_extension
        )
    else:
        save_filename = '{}_{}.{}'.format(
            filename_prefix,
            timestamp.strftime(filename_datetime_format),
            filename_extension
        )
    df = df.copy()
    if min_score is not None:
        df.loc[df['score'] < min_score, 'score'] = np.nan
    if max_score is not None:
        df.loc[df['score'] > max_score, 'score'] = np.nan
    if pd.isnull(df['score']).all():
        logger.warn('No pose pairs meet score criteria')
        return
    df = df.sort_index()
    pose_2d_ids_a = df.index.get_level_values('pose_2d_id_a').unique().sort_values().tolist()
    pose_2d_ids_b = df.index.get_level_values('pose_2d_id_b').unique().sort_values().tolist()
    if pose_label_map_a is None:
        if 'track_label_a' in df.columns:
            track_labels_a = df['track_label_a'].reset_index(level='pose_2d_id_b', drop=True).drop_duplicates()
            pose_labels_a = [track_labels_a.loc[pose_2d_id] for pose_2d_id in pose_2d_ids_a]
        elif len(pose_2d_ids_a) <= 26:
            pose_labels_a = string.ascii_uppercase[:len(pose_2d_ids_a)]
        else:
            pose_labels_a = range(len(pose_2d_ids_a))
        pose_label_map_a = dict(zip(pose_2d_ids_a, pose_labels_a))
    if pose_label_map_b is None:
        if 'track_label_b' in df.columns:
            track_labels_b = df['track_label_b'].reset_index(level='pose_2d_id_a', drop=True).drop_duplicates()
            pose_labels_b = [track_labels_b.loc[pose_2d_id] for pose_2d_id in pose_2d_ids_b]
        elif len(pose_2d_ids_b) <= 26:
            pose_labels_b = string.ascii_uppercase[:len(pose_2d_ids_b)]
        else:
            pose_labels_b = range(len(pose_2d_ids_b))
        pose_label_map_b = dict(zip(pose_2d_ids_b, pose_labels_b))
    scores_df=df.reset_index().pivot(index='pose_2d_id_a', columns='pose_2d_id_b', values='score')
    scores_df.rename(index=pose_label_map_a, inplace=True)
    scores_df.rename(columns=pose_label_map_b, inplace=True)
    scores_df.sort_index(axis=0, inplace=True)
    scores_df.sort_index(axis=1, inplace=True)
    matches_df=df.reset_index().pivot(index='pose_2d_id_a', columns='pose_2d_id_b', values='match')
    matches_df.rename(index=pose_label_map_a, inplace=True)
    matches_df.rename(columns=pose_label_map_b, inplace=True)
    matches_df.sort_index(axis=0, inplace=True)
    matches_df.sort_index(axis=1, inplace=True)
    if color_rule == 'score':
        data = scores_df
        cmap = score_color_map_name
        annot=True
        cbar = True
        cbar_kws = {
            'label': 'Pose pair score'
        }
    elif color_rule == 'match':
        data = matches_df
        annot = scores_df
        cmap = matplotlib.colors.ListedColormap(['lightgray', 'lightgreen'])
        cbar = False
        cbar_kws = None
    else:
        raise ValueError('Color rule \'{}\' not recognized'.format(color_rule))
    ax = sns_plot = sns.heatmap(
        data=data,
        cmap=cmap,
        linewidths=0.1,
        linecolor='gray',
        annot=annot,
        mask=scores_df.isnull(),
        square=True,
        cbar=cbar,
        cbar_kws = cbar_kws
    )
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    if display_camera_names:
        plt.xlabel(camera_names[camera_id_b])
        plt.ylabel(camera_names[camera_id_a])
    else:
        plt.xlabel(camera_id_b)
        plt.ylabel(camera_id_a)
    ax.set_title(ax_title)
    fig = sns_plot.get_figure()
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    # Show plot
    if show:
        plt.show()
    # Save plot
    if save:
        path = os.path.join(
            save_directory,
            save_filename
        )
        fig.savefig(path)

def visualize_poses_3d_top_down_timestamp(
    df,
    room_boundaries = None,
    edge_threshold = None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S.%f',
    pose_label_background_color='red',
    pose_label_background_alpha=0.5,
    pose_label_color='white',
    pose_label_boxstyle='circle',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='poses_3d_top_down',
    filename_datetime_format='%Y%m%d_%H%M%S_%f',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    timestamps = df['timestamp'].unique()
    if len(timestamps) > 1:
        raise ValueError('Multiple timestamps found in data')
    timestamp = timestamps[0]
    if edge_threshold is not None:
        ax_title = '{} ({} edges)'.format(
            timestamp.strftime(plot_title_datetime_format),
            edge_threshold
        )
    else:
        ax_title = timestamp.strftime(plot_title_datetime_format)
    if edge_threshold is not None:
        save_filename = '{}_{}_{}_edges.{}'.format(
            filename_prefix,
            timestamp.strftime(filename_datetime_format),
            edge_threshold,
            filename_extension
        )
    else:
        save_filename = '{}_{}.{}'.format(
            filename_prefix,
            timestamp.strftime(filename_datetime_format),
            filename_extension
        )
    df_group_matches = df.loc[df['group_match']]
    match_group_labels = df['match_group_label'].dropna().unique()
    num_match_groups = len(match_group_labels)
    color_mapping = generate_color_mapping(match_group_labels)
    fig, ax = plt.subplots()
    for match_group_label in match_group_labels:
        poses_3d = np.stack(df_group_matches.loc[
            df_group_matches['match_group_label'] == match_group_label,
            'keypoint_coordinates_3d'
        ])
        centroids_3d = np.nanmedian(poses_3d, axis=1)
        for centroid_3d_index in range(centroids_3d.shape[0]):
            ax.text(
                centroids_3d[centroid_3d_index, 0],
                centroids_3d[centroid_3d_index, 1],
                match_group_label,
                color=pose_label_color,
                bbox={
                    'alpha': pose_label_background_alpha,
                    'facecolor': color_mapping[match_group_label],
                    'edgecolor': 'none',
                    'boxstyle': pose_label_boxstyle
                }
            )
    if room_boundaries is not None:
        ax.set_xlim(room_boundaries[0][0], room_boundaries[0][1])
        ax.set_ylim(room_boundaries[1][0], room_boundaries[1][1])
    ax.set_xlabel('$x$ (meters)')
    ax.set_ylabel('$y$ (meters)')
    ax.set_aspect('equal')
    ax.set_title(ax_title)
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    # Show plot
    if show:
        plt.show()
    # Save plot
    if save:
        path = os.path.join(
            save_directory,
            save_filename
        )
        fig.savefig(path)

def pose_track_3d_timelines_by_camera(
    df,
    color_by_pose_quality=False,
    display_camera_name=False,
    camera_names=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='pose_track_3d_timelines',
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if display_camera_name:
        if camera_names is None:
            camera_ids = df['camera_id'].unique().tolist()
            camera_names = honeycomb_io.fetch_camera_names(camera_ids)
    for camera_id, group_df in df.groupby('camera_id'):
        if display_camera_name:
            camera_id_string = camera_names.get(camera_id)
        else:
            camera_id_string = camera_id
        plot_title = camera_id_string
        file_identifier = camera_id_string
        pose_track_3d_timelines(
            df=group_df,
            color_by_pose_quality=color_by_pose_quality,
            plot_title=plot_title,
            plot_title_datetime_format=plot_title_datetime_format,
            show=show,
            save=save,
            save_directory=save_directory,
            filename_prefix=filename_prefix,
            file_identifier=file_identifier,
            filename_datetime_format=filename_datetime_format,
            filename_extension=filename_extension,
            fig_width_inches=fig_width_inches,
            fig_height_inches=fig_height_inches
        )

def pose_track_3d_timelines(
    df,
    color_by_pose_quality=False,
    plot_title=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='pose_track_3d_timelines',
    file_identifier=None,
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if plot_title is not None:
        fig_suptitle = '{} ({} - {})'.format(
            plot_title,
            df['timestamp'].min().strftime(plot_title_datetime_format),
            df['timestamp'].max().strftime(plot_title_datetime_format)
        )
    else:
        fig_suptitle = '{} - {}'.format(
            df['timestamp'].min().strftime(plot_title_datetime_format),
            df['timestamp'].max().strftime(plot_title_datetime_format)
        )
    if file_identifier is not None:
        save_filename = '{}_{}_{}_{}.{}'.format(
            filename_prefix,
            slugify.slugify(file_identifier),
            df['timestamp'].min().strftime(filename_datetime_format),
            df['timestamp'].max().strftime(filename_datetime_format),
            filename_extension
        )
    else:
        save_filename = '{}_{}_{}.{}'.format(
            filename_prefix,
            df['timestamp'].min().strftime(filename_datetime_format),
            df['timestamp'].max().strftime(filename_datetime_format),
            filename_extension
        )
    if color_by_pose_quality:
        hue=df['pose_quality_2d']
    else:
        hue=None
    sns_plot = sns.scatterplot(
        x=df['timestamp'],
        y=df['track_label'],
        hue=hue
    )
    sns_plot.set_xlim(
        df['timestamp'].min(),
        df['timestamp'].max()
    )
    plt.xlabel('Time')
    plt.ylabel('Track label')
    fig = sns_plot.get_figure()
    fig.suptitle(fig_suptitle)
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    # Show plot
    if show:
        plt.show()
    # Save plot
    if save:
        path = os.path.join(
            save_directory,
            save_filename
        )
        fig.savefig(path)

def generate_color_mapping(labels):
    colors = sns.color_palette('husl', n_colors=len(labels))
    color_mapping = dict(zip(sorted(labels), colors))
    return color_mapping

def generate_color_mapping_no_sort(labels):
    colors = sns.color_palette('husl', n_colors=len(labels))
    color_mapping = dict(zip(labels, colors))
    return color_mapping
