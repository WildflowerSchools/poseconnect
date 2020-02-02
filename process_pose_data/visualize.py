import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
import slugify
import os

register_matplotlib_converters()

def track_timelines(
    df,
    show=True,
    save=False,
    pose_quality_colormap_name='summer_r',
    save_directory='.',
    filename_suffix='_track_timeline',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    for camera_device_id, group_df in df.groupby('camera_device_id'):
        track_timeline(
            df=group_df,
            show=show,
            save=save,
            pose_quality_colormap_name=pose_quality_colormap_name,
            save_directory=save_directory,
            filename_suffix=filename_suffix,
            filename_extension=filename_extension,
            fig_width_inches=fig_width_inches,
            fig_height_inches=fig_height_inches
        )

def track_timeline(
    df,
    show=True,
    save=False,
    pose_quality_colormap_name='summer_r',
    save_directory='.',
    filename_suffix='_track_timeline',
    filename_extension='png',
    fig_width_inches=10.5,
    fig_height_inches=8
):
    # Extract camera info
    camera_info = extract_camera_info(df)
    # Convert track labels to integers if possible
    try:
        track_labels=df['track_label'].astype('int')
    except:
        track_labels=df['track_label']
    # Build plot
    fig, axes = plt.subplots()
    plot_object=axes.scatter(
        # df['timestamp'].tz_convert(None),
        df['timestamp'],
        track_labels,
        c=df['pose_quality'],
        cmap=plt.get_cmap(pose_quality_colormap_name),
        marker='.'
    )
    axes.set_xlim(
        df['timestamp'].min(),
        df['timestamp'].max()
    )
    axes.set_xlabel('Time (UTC)')
    axes.set_ylabel('Track label')
    fig.colorbar(plot_object, ax=axes).set_label('Pose quality')
    fig.suptitle('{} ({})'.format(
        camera_info['camera_name'],
        df['timestamp'].min().isoformat()))
    fig.set_size_inches(fig_width_inches, fig_height_inches)
    # Show plot
    if show:
        plt.show()
    # Save plot
    if save:
        path = os.path.join(
            save_directory,
            '{}{}.{}'.format(
                slugify.slugify(camera_info['camera_name']),
                filename_suffix,
                filename_extension
            )
        )
        fig.savefig(path)

def extract_camera_info(df):
    # Extract camera device ID
    camera_device_ids = df['camera_device_id'].unique().tolist()
    if len(camera_device_ids) > 1:
        raise ValueError('Data contains more than one camera device ID: {}'.format(
            camera_device_ids
        ))
    camera_device_id = camera_device_ids[0]
    # Extract camera name
    camera_names = df['camera_name'].unique().tolist()
    if len(camera_names) > 1:
        raise ValueError('Data contains more than one camera name: {}'.format(
            camera_names
        ))
    camera_name = camera_names[0]
    return {
        'camera_device_id': camera_device_id,
        'camera_name': camera_name
    }
