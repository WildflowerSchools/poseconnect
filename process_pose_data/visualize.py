import process_pose_data.fetch
import cv_utils
import cv2 as cv
import video_io
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
            camera_names = process_pose_data.fetch.fetch_camera_names(camera_ids)
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
        np.concatenate(df['keypoint_quality'].values),
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
            camera_names = process_pose_data.fetch.fetch_camera_names(camera_ids)
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
    num_valid_keypoints_series = df['keypoint_quality'].apply(lambda x: np.count_nonzero(~np.isnan(x)))
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
            camera_names = process_pose_data.fetch.fetch_camera_names(camera_ids)
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
        df['pose_quality'],
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
            camera_names = process_pose_data.fetch.fetch_camera_names(camera_ids)
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
        df['keypoint_quality'].apply(lambda x: np.nanmean(x)),
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
            camera_names = process_pose_data.fetch.fetch_camera_names(camera_ids)
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
        x=df['keypoint_quality'].apply(lambda x: np.nanmean(x)),
        y=df['pose_quality']
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
            camera_names = process_pose_data.fetch.fetch_camera_names(camera_ids)
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
    pose_ids_a = df.index.get_level_values('pose_id_a').unique().sort_values().tolist()
    pose_ids_b = df.index.get_level_values('pose_id_b').unique().sort_values().tolist()
    if pose_label_map_a is None:
        if 'track_label_a' in df.columns:
            track_labels_a = df['track_label_a'].reset_index(level='pose_id_b', drop=True).drop_duplicates()
            pose_labels_a = [track_labels_a.loc[pose_id] for pose_id in pose_ids_a]
        elif len(pose_ids_a) <= 26:
            pose_labels_a = string.ascii_uppercase[:len(pose_ids_a)]
        else:
            pose_labels_a = range(len(pose_ids_a))
        pose_label_map_a = dict(zip(pose_ids_a, pose_labels_a))
    if pose_label_map_b is None:
        if 'track_label_b' in df.columns:
            track_labels_b = df['track_label_b'].reset_index(level='pose_id_a', drop=True).drop_duplicates()
            pose_labels_b = [track_labels_b.loc[pose_id] for pose_id in pose_ids_b]
        elif len(pose_ids_b) <= 26:
            pose_labels_b = string.ascii_uppercase[:len(pose_ids_b)]
        else:
            pose_labels_b = range(len(pose_ids_b))
        pose_label_map_b = dict(zip(pose_ids_b, pose_labels_b))
    scores_df=df.reset_index().pivot(index='pose_id_a', columns='pose_id_b', values='score')
    scores_df.rename(index=pose_label_map_a, inplace=True)
    scores_df.rename(columns=pose_label_map_b, inplace=True)
    scores_df.sort_index(axis=0, inplace=True)
    scores_df.sort_index(axis=1, inplace=True)
    matches_df=df.reset_index().pivot(index='pose_id_a', columns='pose_id_b', values='match')
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

def extract_single_camera_data(
    df,
):
    single_camera_columns = list()
    for column in df.columns:
        if column[-2:]=='_a' or column[-2:]=='_b':
            single_camera_column = column[:-2]
            if single_camera_column not in single_camera_columns:
                single_camera_columns.append(single_camera_column)
    df = df.reset_index()
    dfs = dict()
    for camera_letter in ['a', 'b']:
        extraction_columns = ['pose_id_' + camera_letter, 'timestamp']
        extraction_columns.extend([single_camera_column + '_' + camera_letter for single_camera_column in single_camera_columns])
        target_columns = ['pose_id', 'timestamp']
        target_columns.extend(single_camera_columns)
        column_map = dict(zip(extraction_columns, target_columns))
        df_single_camera = df.reindex(columns=extraction_columns)
        df_single_camera.rename(columns=column_map, inplace=True)
        df_single_camera.drop_duplicates(subset='pose_id', inplace=True)
        df_single_camera.set_index('pose_id', inplace=True)
        dfs[camera_letter] = df_single_camera
    return dfs

def draw_poses_2d_timestamp_camera_pair(
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
    pose_label_boxstyle='circle',
    image_size=None,
    show_axes=False,
    show_background_image=True,
    background_image=None,
    background_image_alpha=0.4,
    display_camera_name=False,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S.%f',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='poses_2d',
    filename_datetime_format='%Y%m%d_%H%M%S_%f',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    dfs_single_camera = extract_single_camera_data(df)
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
        draw_poses_2d_timestamp_camera(
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
            pose_label_boxstyle=pose_label_boxstyle,
            image_size=image_size,
            show_axes=show_axes,
            show_background_image=show_background_image,
            background_image=background_image,
            background_image_alpha=background_image_alpha,
            display_camera_name=display_camera_name,
            camera_name=camera_names[camera_letter],
            plot_title_datetime_format=plot_title_datetime_format,
            show=show,
            save=save,
            save_directory=save_directory,
            filename_prefix=filename_prefix,
            filename_datetime_format=filename_datetime_format,
            filename_extension=filename_extension,
            fig_width_inches=fig_width_inches,
            fig_height_inches=fig_height_inches
        )

def draw_poses_2d_timestamp_camera(
    df,
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    pose_label_map=None,
    pose_color_map=None,
    keypoint_alpha=0.3,
    keypoint_connector_alpha=0.3,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.5,
    pose_label_boxstyle='circle',
    image_size=None,
    show_axes=False,
    show_background_image=True,
    background_image=None,
    background_image_alpha=0.4,
    display_camera_name=False,
    camera_name=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S.%f',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='poses_2d',
    filename_datetime_format='%Y%m%d_%H%M%S_%f',
    filename_extension='png',
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
    if display_camera_name:
        if camera_name is None:
            camera_name = process_pose_data.fetch.fetch_camera_names([camera_id])[camera_id]
        camera_identifier = camera_name
    fig_suptitle = '{} ({})'.format(
        camera_identifier,
        timestamp.strftime(plot_title_datetime_format)
    )
    save_filename = '{}_{}_{}.{}'.format(
        filename_prefix,
        slugify.slugify(camera_identifier),
        timestamp.strftime(filename_datetime_format),
        filename_extension
    )
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
    for pose_id, row in df.iterrows():
        process_pose_data.draw_pose_2d(
            row['keypoint_coordinates'],
            draw_keypoint_connectors=draw_keypoint_connectors,
            keypoint_connectors=keypoint_connectors,
            pose_label=pose_label_map[pose_id],
            pose_color=pose_color_map[pose_id],
            keypoint_alpha=keypoint_alpha,
            keypoint_connector_alpha=keypoint_connector_alpha,
            keypoint_connector_linewidth=keypoint_connector_linewidth,
            pose_label_color=pose_label_color,
            pose_label_background_alpha=pose_label_background_alpha,
            pose_label_boxstyle=pose_label_boxstyle
        )
    if show_background_image:
        if background_image is None:
            image_metadata = video_io.fetch_images(
                image_timestamps = [timestamp.to_pydatetime()],
                camera_device_ids=[camera_id]
            )
            image_local_path = image_metadata[0]['image_local_path']
            background_image = cv_utils.fetch_image_from_local_drive(image_local_path)
        cv_utils.draw_background_image(
            image=background_image,
            alpha=background_image_alpha
        )
    cv_utils.format_2d_image_plot(
        image_size=image_size,
        show_axes=show_axes
    )
    fig = plt.gcf()
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

def draw_pose_2d(
    keypoint_coordinates,
    draw_keypoint_connectors=True,
    keypoint_connectors=None,
    pose_label=None,
    pose_color=None,
    keypoint_alpha=0.3,
    keypoint_connector_alpha=0.3,
    keypoint_connector_linewidth=3,
    pose_label_color='white',
    pose_label_background_alpha=0.5,
    pose_label_boxstyle='circle'
):
    keypoint_coordinates = np.asarray(keypoint_coordinates).reshape((-1, 2))
    valid_keypoints = np.all(np.isfinite(keypoint_coordinates), axis=1)
    plottable_points = keypoint_coordinates[valid_keypoints]
    points_image_u = plottable_points[:, 0]
    points_image_v = plottable_points[:, 1]
    plot, = plt.plot(
        points_image_u,
        points_image_v,
        '.',
        color=pose_color,
        alpha=keypoint_alpha)
    plot_color=color=plot.get_color()
    if draw_keypoint_connectors and (keypoint_connectors is not None):
        for keypoint_connector in keypoint_connectors:
            keypoint_from_index = keypoint_connector[0]
            keypoint_to_index = keypoint_connector[1]
            if valid_keypoints[keypoint_from_index] and valid_keypoints[keypoint_to_index]:
                plt.plot(
                    [keypoint_coordinates[keypoint_from_index,0],keypoint_coordinates[keypoint_to_index, 0]],
                    [keypoint_coordinates[keypoint_from_index,1],keypoint_coordinates[keypoint_to_index, 1]],
                    linewidth=keypoint_connector_linewidth,
                    color=plot_color,
                    alpha=keypoint_connector_alpha
                )
    if pose_label is not None:
        pose_label_anchor = np.nanmean(keypoint_coordinates, axis=0)
        plt.text(
            pose_label_anchor[0],
            pose_label_anchor[1],
            pose_label,
            color=pose_label_color,
            bbox={
                'alpha': pose_label_background_alpha,
                'facecolor': plot_color,
                'edgecolor': 'none',
                'boxstyle': pose_label_boxstyle
            }
        )

def visualize_pose_pair(
    pose_pair,
    camera_calibrations=None,
    camera_names=None,
    floor_marker_color='blue',
    floor_marker_linewidth=2,
    floor_marker_linestyle='-',
    floor_marker_alpha=1.0,
    vertical_line_color='blue',
    vertical_line_linewidth=2,
    vertical_line_linestyle='--',
    vertical_line_alpha=1.0,
    centroid_markersize=16,
    centroid_color='red',
    centroid_alpha=1.0,
    pose_label_background_color='red',
    pose_label_background_alpha=1.0,
    pose_label_color='white',
    pose_label_boxstyle='circle',
    background_image_alpha=0.4,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S.%f',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='pose_pair',
    filename_datetime_format='%Y%m%d_%H%M%S_%f',
    filename_extension='png',
    fig_width_inches=8,
    fig_height_inches=10.5
):
    timestamp = pose_pair.get('timestamp')
    camera_id_a = pose_pair.get('camera_id_a')
    camera_id_b = pose_pair.get('camera_id_b')
    if camera_calibrations is None:
        camera_calibrations = process_pose_data.fetch_camera_calibrations(
            camera_ids=[camera_id_a, camera_id_b],
            start=timestamp.to_pydatetime(),
            end=timestamp.to_pydatetime()
        )
    if camera_names is None:
        camera_names = process_pose_data.fetch_camera_names(
            camera_ids=[camera_id_a, camera_id_b]
        )
    fig_suptitle = timestamp.strftime(plot_title_datetime_format)
    save_filename = '{}_{}_{}.{}'.format(
        filename_prefix,
        slugify.slugify(pose_pair['pose_id_a']),
        slugify.slugify(pose_pair['pose_id_b']),
        filename_extension
    )
    centroid_3d = np.nanmean(pose_pair['keypoint_coordinates_3d'], axis=0)
    floor_marker_x_3d = np.array([[x, centroid_3d[1], 0] for x in np.linspace(centroid_3d[0] - 1.0, centroid_3d[0] + 1.0, 100)])
    floor_marker_y_3d = np.array([[centroid_3d[0], y, 0] for y in np.linspace(centroid_3d[1] - 1.0, centroid_3d[1] + 1.0, 100)])
    vertical_line_3d = np.array([[centroid_3d[0], centroid_3d[1], z] for z in np.linspace(0, centroid_3d[2], 100)])
    fig, axes = plt.subplots(2, 1)
    for axis_index, suffix in enumerate(['a', 'b']):
        axis_title = camera_names[pose_pair['camera_id_' + suffix]]
        centroid = cv_utils.project_points(
            object_points=centroid_3d,
            rotation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['rotation_vector'],
            translation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['translation_vector'],
            camera_matrix=camera_calibrations[pose_pair['camera_id_' + suffix]]['camera_matrix'],
            distortion_coefficients=camera_calibrations[pose_pair['camera_id_' + suffix]]['distortion_coefficients'],
            remove_behind_camera=True
        )
        floor_marker_x = cv_utils.project_points(
            object_points=floor_marker_x_3d,
            rotation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['rotation_vector'],
            translation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['translation_vector'],
            camera_matrix=camera_calibrations[pose_pair['camera_id_' + suffix]]['camera_matrix'],
            distortion_coefficients=camera_calibrations[pose_pair['camera_id_' + suffix]]['distortion_coefficients'],
            remove_behind_camera=True
        )
        floor_marker_y = cv_utils.project_points(
            object_points=floor_marker_y_3d,
            rotation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['rotation_vector'],
            translation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['translation_vector'],
            camera_matrix=camera_calibrations[pose_pair['camera_id_' + suffix]]['camera_matrix'],
            distortion_coefficients=camera_calibrations[pose_pair['camera_id_' + suffix]]['distortion_coefficients'],
            remove_behind_camera=True
        )
        vertical_line = cv_utils.project_points(
            object_points=vertical_line_3d,
            rotation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['rotation_vector'],
            translation_vector=camera_calibrations[pose_pair['camera_id_' + suffix]]['translation_vector'],
            camera_matrix=camera_calibrations[pose_pair['camera_id_' + suffix]]['camera_matrix'],
            distortion_coefficients=camera_calibrations[pose_pair['camera_id_' + suffix]]['distortion_coefficients'],
            remove_behind_camera=True
        )
        image_metadata = video_io.fetch_images(
            image_timestamps = [timestamp.to_pydatetime()],
            camera_device_ids=[pose_pair['camera_id_' + suffix]]
        )
        image_local_path = image_metadata[0]['image_local_path']
        background_image = cv_utils.fetch_image_from_local_drive(image_local_path)
        axes[axis_index].imshow(
            cv.cvtColor(background_image, cv.COLOR_BGR2RGB),
            alpha=background_image_alpha
        )
        if pose_pair.get('track_label_' + suffix) is not None:
            axes[axis_index].text(
                centroid[0],
                centroid[1],
                pose_pair.get('track_label_' + suffix),
                color=pose_label_color,
                bbox={
                    'alpha': pose_label_background_alpha,
                    'facecolor': pose_label_background_color,
                    'edgecolor': 'none',
                    'boxstyle': pose_label_boxstyle
                }
            )
        else:
            plt.plot(
                centroid[0],
                centroid[1],
                '.',
                color=centroid_color,
                markersize=centroid_marker_size,
                alpha=centroid_alpha
            )
        axes[axis_index].plot(
            floor_marker_x[:, 0],
            floor_marker_x[:, 1],
            color=floor_marker_color,
            linewidth=floor_marker_linewidth,
            linestyle=floor_marker_linestyle,
            alpha=floor_marker_alpha,
        )
        axes[axis_index].plot(
            floor_marker_y[:, 0],
            floor_marker_y[:, 1],
            color=floor_marker_color,
            linewidth=floor_marker_linewidth,
            linestyle=floor_marker_linestyle,
            alpha=floor_marker_alpha,
        )
        axes[axis_index].plot(
            vertical_line[:, 0],
            vertical_line[:, 1],
            color=vertical_line_color,
            linewidth=vertical_line_linewidth,
            linestyle=vertical_line_linestyle,
            alpha=vertical_line_alpha,
        )
        axes[axis_index].axis(
            xmin=0,
            xmax=camera_calibrations[pose_pair['camera_id_' + suffix]]['image_width'],
            ymin=camera_calibrations[pose_pair['camera_id_' + suffix]]['image_height'],
            ymax=0
        )
        axes[axis_index].axis('off')
        axes[axis_index].set_title(axis_title)
    fig.suptitle(fig_suptitle)
    plt.subplots_adjust(top=0.95, hspace=0.1)
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    plt.show()
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

def pose_track_timelines_by_camera(
    df,
    color_by_pose_quality=False,
    display_camera_name=False,
    camera_names=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='pose_track_timelines',
    filename_datetime_format='%Y%m%d_%H%M%S',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    if display_camera_name:
        if camera_names is None:
            camera_ids = df['camera_id'].unique().tolist()
            camera_names = process_pose_data.fetch.fetch_camera_names(camera_ids)
    for camera_id, group_df in df.groupby('camera_id'):
        if display_camera_name:
            camera_id_string = camera_names.get(camera_id)
        else:
            camera_id_string = camera_id
        plot_title = camera_id_string
        file_identifier = camera_id_string
        pose_track_timelines(
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

def pose_track_timelines(
    df,
    color_by_pose_quality=False,
    plot_title=None,
    plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
    show=True,
    save=False,
    save_directory='.',
    filename_prefix='pose_track_timelines',
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
        hue=df['pose_quality']
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
