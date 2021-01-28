# process_pose_data

Tools for fetching, processing, visualizing, and analyzing Wildflower human pose data

## Task list

* Diagnose missing CUWB data in analysis of 19:48-19:55 UTC on Jan 21
* Diagnose floating pose in output from 19:48-19:55 UTC on Jan 21 and from earlier test night
* Add ability to write locally generated object IDs to Honeycomb
* Create separate workers for 3D pose reconstruction, 3D pose tracking, 3D pose track interpolation, and 3D pose track identification (saving output from each stage locally)
* Create separate workers for uploading to Honeycomb 3D poses, 3D pose tracks, interpolated 3D pose tracks, 3D pose identification, 3D pose track identification
* Clean up argument ordering in `reconstruct_poses_3d_alphapose_local_time_segment`
* Retool `generate_inference_metadata_reconstruct_3d_poses_alphapose_local` to exclude cameras without calibration data
* Make function to delete Honeycomb inference executions
* Make function to delete local inference metadata
* Make function to delete local 3D pose files
* Update `honeycomb_io.write_3d_pose_data` to incorporate inference execution ID
* Update `honeycomb_io.fetch_3d_pose_data` to incorporate inference execution ID
* Make function for deleting local 3D pose data
* Make entry point for batch writing 3D pose data to Honeycomb
* Make pipeline function for generating pose tracks and interpolating pose tracks
* Update `honeycomb_io.write_3d_pose_tracks` to write interpolated 3D pose tracks to Honeycomb
* Make pipeline function for identifying 3D pose tracks
* Make function for writing pose track identification inferences to Honeycomb
* Figure out why there's an empty `pose_id_3d_local` column in 3D pose dataframe output
* Fix up overlay functions so they cope if overlay already exists
* Fix up 3D reconstruction pipeline so Honeycomb client info flows properly through function stack
* Figure out how to make code source package version number
* Add `poses_2d_json_format` option to `reconstruct_poses_3d` entry point
* Parallellize 3D pose overlay function
* Rewrite all log messages so formatting isn't called if log isn't printed
* Rewrite `overlay.overlay_video_poses_2d()` to match functionality of `overlay.overlay_video_poses_3d()` (e.g., more flexible specification of videos, concatenation)
* Extend ability to set output container and code to all overlay functions
* Loosen checks on overlap between pose data and video data (go ahead as long as there is _some_ overlap)
* Add ability to filter 3D pose tracks on track duration/size
* Add option to overlay function to first filter on 3D pose track duration/size
* Add ability to overlay for a time range (batch processing)
* Add option to concatenate videos across time range for each camera
* Dockerize pipeline
* Set up pipeline for Airflow
* Add function to fetch pose tracks from Honeycomb
* Add progress bar option to `generate_pose_tracks()`
* Make functions handle empty poses (all keypoints `NaN`) more gracefully (e.g., `score_pose_pairs()`, `draw_pose_2d()`)
* Fix up `honeycomb_io` module to match design of `wf-video-io`
  - Allow user to supply Honeycomb client
  - Clean up default setting
  - Other?
  * Remove unused functions and methods
* Make visualization functions handle missing fields (e.g., `pose_quality`) more gracefully
* Figure out inconsistent behavior of `groupby(...).apply(...)` (under what conditions does it add grouping variables to index?)
* For functions that act on dataframes, make it optional to check dataframe structure (e.g., only one timestamp and camera pair)
* For functions than iterate over previous functions, making naming and approach consistent (e.g., always use apply?)
* Add `keypoint_categories` info to pose models in Honeycomb?
* Be consistent about whether to convert track labels to integers (where possible)
* Remove dependence on OpenCV by adding necessary functionality to `cv_utils`
* Consider refactoring split between `video_io` and `cv_utils`
* Fix up `cv_utils` Matplotlib drawing functions so that they accept an axis (or figure, as appropriate)
* Fix up handling of background image alpha (shouldn't assume white background)
* Fix up _y_ axis inversion for images (go back to `cv_utils`?)
* Add option of specifying Honeycomb client info for visualization functions that require Honeycomb
* Reinstate `sns.set()` for Seaborn plots without making it spill over into non-Seaborn plots (see [here](https://stackoverflow.com/questions/26899310/python-seaborn-to-reset-back-to-the-matplotlib))
* Refactor code in `visualize` to make it less repetitive (same pattern over and over for `[verb]_by_camera`)
* Fix up legend on pose track timelines
* Add visualization for number of poses per camera per timestamp
* Replace `cv.triangulatePoints()` to increase speed (and hopefully accuracy)
* Get pose video overlays working again (for data with track labels)
* Rewrite geom rendering functions to handle the possibility of no track labels
* Rewrite function which overlays geoms on videos so that user can specify a time span that it is a subset of the geoms and/or the video
* Make all time inputs more permissive (in terms of type/format) and make all time outputs more consistent
* Be consistent about accepting timestamp arguments in any format parseable by `pd.to_datetime()`
