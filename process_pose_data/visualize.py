import process_pose_data.fetch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import slugify
import os

sns.set()

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

# def pose_pair_score_histogram_by_camera(
#     df,
#     display_camera_name=False,
#     camera_names=None,
#     bins=None,
#     plot_title_datetime_format='%m/%d/%Y %H:%M:%S',
#     show=True,
#     save=False,
#     save_directory='.',
#     filename_prefix='pose_pair_score',
#     filename_datetime_format='%Y%m%d_%H%M%S',
#     filename_extension='png',
#     fig_width_inches=10.5,
#     fig_height_inches=8
# ):
#     if display_camera_name:
#         if camera_names is None:
#             camera_ids = df['camera_id'].unique().tolist()
#             camera_names = process_pose_data.fetch.fetch_camera_names(camera_ids)
#     for camera_id, group_df in df.groupby('camera_id'):
#         if display_camera_name:
#             camera_id_string = camera_names.get(camera_id)
#         else:
#             camera_id_string = camera_id
#         plot_title = camera_id_string
#         file_identifier = camera_id_string
#         pose_quality_histogram(
#             df=group_df,
#             bins=bins,
#             plot_title=plot_title,
#             plot_title_datetime_format=plot_title_datetime_format,
#             show=show,
#             save=save,
#             save_directory=save_directory,
#             filename_prefix=filename_prefix,
#             file_identifier=file_identifier,
#             filename_datetime_format=filename_datetime_format,
#             filename_extension=filename_extension,
#             fig_width_inches=fig_width_inches,
#             fig_height_inches=fig_height_inches
#         )

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
    timestamp_min = df.index.get_level_values('timestamp').min()
    timestamp_max = df.index.get_level_values('timestamp').max()
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

def pose_pair_score_heatmap(
    df,
    min_score=None,
    max_score=None,
    color_map_name='summer_r',
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
    timestamps = df.index.get_level_values('timestamp').unique()
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
        fig_suptitle = '{} ({})'.format(
            plot_title,
            timestamp.strftime(plot_title_datetime_format)
        )
    else:
        fig_suptitle = '{}'.format(
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
    if min_score is not None:
        df.loc[df['score'] < min_score, 'score'] = np.nan
    if max_score is not None:
        df.loc[df['score'] > max_score, 'score'] = np.nan
    pivot_df=df.reset_index().pivot(index='pose_id_a', columns='pose_id_b', values='score')
    pivot_df.index = np.arange(pivot_df.shape[0])
    pivot_df.columns = np.arange(pivot_df.shape[1])
    sns_plot = sns.heatmap(
        pivot_df,
        cmap=color_map_name,
        linewidths=0.1,
        linecolor='gray',
        annot=True,
        square=True,
        cbar_kws = {
            'label': 'Pose pair score'
        }
    )
    if display_camera_names:
        plt.xlabel(camera_names[camera_id_b])
        plt.ylabel(camera_names[camera_id_a])
    else:
        plt.xlabel(camera_id_b)
        plt.ylabel(camera_id_a)
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

# def pose_track_scores_heatmap(
#     df,
#     camera_device_ids,
#     score_metric,
#     min_num_common_frames=None,
#     min_score_metric=None,
#     max_score_metric=None,
#     color_map_name = 'summer_r',
#     score_axis_label = 'Score',
#     title_label = 'Scores',
#     show=True,
#     save=False,
#     save_directory='.',
#     filename='match_scores_heatmap',
#     filename_extension='png',
#     fig_width_inches=10.5,
#     fig_height_inches=8
# ):
#     if len(camera_device_ids) != 2:
#         raise ValueError('Must specify exactly two camera device IDs')
#     camera_device_ids_a = df['camera_device_id_a'].unique().tolist()
#     camera_device_ids_b = df['camera_device_id_b'].unique().tolist()
#     if camera_device_ids[0] in camera_device_ids_a and camera_device_ids[1] in camera_device_ids_b:
#         camera_device_id_a = camera_device_ids[0]
#         camera_device_id_b = camera_device_ids[1]
#     elif camera_device_ids[1] in camera_device_ids_a and camera_device_ids[0] in camera_device_ids_b:
#         camera_device_id_a = camera_device_ids[1]
#         camera_device_id_b = camera_device_ids[0]
#     else:
#         raise ValueError('Camera pair not found in data')
#     scores_df = df.loc[
#         (df['camera_device_id_a'] == camera_device_id_a) &
#         (df['camera_device_id_b'] == camera_device_id_b)
#     ].copy()
#     camera_names_a = scores_df['camera_name_a'].unique().tolist()
#     camera_names_b = scores_df['camera_name_b'].unique().tolist()
#     if len(camera_names_a) > 1:
#         raise ValueError('More than one camera name found for camera A')
#     if len(camera_names_b) > 1:
#         raise ValueError('More than one camera name found for camera B')
#     camera_name_a = camera_names_a[0]
#     camera_name_b = camera_names_b[0]
#     df = scores_df.copy()
#     if min_num_common_frames is not None:
#         df.loc[df['num_common_frames'] < min_num_common_frames, score_metric] = np.nan
#     if min_score_metric is not None:
#         df.loc[df[score_metric] < min_score_metric, score_metric] = np.nan
#     if max_score_metric is not None:
#         df.loc[df[score_metric] > max_score_metric, score_metric] = np.nan
#     pivot_df=df.pivot(index='track_label_a', columns='track_label_b', values=score_metric)
#     fig, axes = plt.subplots()
#     sns.heatmap(
#         pivot_df,
#         cmap=color_map_name,
#         linewidths=0.1,
#         linecolor='gray',
#         annot=True,
#         ax=axes,
#         square=True,
#         cbar_kws = {
#             'label': score_axis_label
#         }
#     )
#     axes.set_ylabel('{} track labels'.format(camera_name_a))
#     axes.set_xlabel('{} track labels'.format(camera_name_b))
#     axes.set_title(title_label)
#     fig.set_size_inches(fig_width_inches, fig_height_inches)
#     # Show plot
#     if show:
#         plt.show()
#     # Save plot
#     if save:
#         path = os.path.join(
#             save_directory,
#             '{}.{}'.format(
#                 filename,
#                 filename_extension
#             )
#         )
#         fig.savefig(path)
