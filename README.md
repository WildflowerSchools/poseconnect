# process_pose_data

Tools for fetching, processing, visualizing, and analyzing Wildflower human pose data

## Task list

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
