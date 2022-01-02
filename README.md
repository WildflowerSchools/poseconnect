# poseconnect

Tools for constructing 3D pose tracks from multi-camera 2D poses

## Installation

To install both the Python library and the command line interface, launch a
Python virtual environment (optional, but recommended) and then install via
`pip`:

```
pip install poseconnect
```

## Command line interface

The command line interface consists of a single command (`poseconnect`) with
four subcommands:

* `reconstruct` takes a file of 2D pose data and a file of camera calibration info as inputs and outputs a file of 3D pose data
* `track` takes a file of 3D pose data as input and outputs a file of 3D pose data with track IDs
* `interpolate` takes a file of 3D pose data with track IDs as input and fills time gaps in each track with linearly interpolated poses
* `identify` takes a file of 3D pose data with track IDs and a file of location sensor data with person IDs and outputs a file of 3D pose data with track and person IDs
* `overlay` takes a file of 2D or 3D pose data and a source video for the data and outputs a video with the poses overlaid

### Help

Overall help is available at `poseconnect --help` and help for each subcommand
(including usage and a list of options) is available at `poseconnect SUBCOMMAND
--help` (e.g., `poseconnect reconstruct --help`).

### Data formats

More information about input and output data file formats is available [here](https://github.com/WildflowerSchools/poseconnect/blob/master/docs/data_formats.md).

### Demo

To see the command line interface in action, download and unzip the following sample JSON data files to your current directory and run the [demo shell script](https://github.com/WildflowerSchools/poseconnect/blob/master/scripts/poseconnect_demo.sh):

* [2D pose data](https://wildflower-tech-public.s3.us-east-2.amazonaws.com/poseconnect/sample_pose_2d_data.zip)
* [Camera calibration info](https://wildflower-tech-public.s3.us-east-2.amazonaws.com/poseconnect/sample_camera_calibration_info.zip)
* [Sensor data](https://wildflower-tech-public.s3.us-east-2.amazonaws.com/poseconnect/sample_sensor_data.zip)
