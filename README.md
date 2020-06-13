# process_pose_data

Tools for fetching, processing, visualizing, and analyzing Wildflower human pose data

## Task list

* Remove profiling logic in `analyze.py`
* Consistently add `_2d` or `_3d` to variable names and field names which could be either
* Consistent add `_local` to IDs that only exist locally (not on Honeycomb)
* Add logic to convert coordinates equal to 0.0 to `np.nan`?
* Consistently set default algorithm parameters to values that seem to have been working the best
* Fetch pose model name from pose model ID rather than making user specify (for 3D pose generation)
* Fetch coordinate space ID rather than making user specify (for writing 3D poses)
* Remove `match_group_label` from 3D pose generation output?
* Add progress bar option to `generate_pose_tracks()`
* Add logic to fetch functions to convert `None` values in float arrays to `np.nan`
* Consolidate `search_[object]s` functions into one function that takes mutation name and id field name as argument
* Reorder functions in `honeycomb_io`
* Comment out all functions and methods that are unused and run entire pipeline in different configurations to verify
* Remove unused functions and methods
* Add logic to `reconstruct_poses_3d()` to fetch camera calibrations if they are not supplied
* Generally make functions more robust to missing calibration data (e.g., issue warning and drop poses rather than stopping)
* Make functions handle empty poses (all keypoints `NaN`) more gracefully (e.g., `score_pose_pairs()`, `draw_pose_2d()`)
* Fix up `fetch` module to match design of `wf-video-io`
  - Allow user to supply Honeycomb client
  - Clean up default setting
  - Other?
* Dockerize pipeline
* Set up pipeline for Airflow
* Make visualization functions handle missing fields (e.g., `pose_quality`) more gracefully
* Add progress bars to all pipeline functions
* Restructure `process_poses_by_timestamp` to use `groupby(...).apply(...)` pattern
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
