# poseconnect

Tools for constructing 3D pose tracks from multi-camera 2D poses

## Task list

* Remove duplicates from sample sensor data
* Add ability to set command line defaults using environment variables
* Add ability to set library defaults using environment variables
* Add ability to specify environment variables using `dotenv`
* Provide better command line UI for which `None` value has specific meaning
* Regularize use of progress bars (everywhere or nowhere)
* Consider removing pose pair score distance method options
* Consider removing pose pair score summary method options
* Add documentation for command line interface
* Add documentation for library interface
* Add documentation for installation
* Add documentation for sample/demo usage
* Add documentation for help functionality
* Add simple video overlay capability
* Add basic batch processing capabilities
* Add basic multiprocessing capabilities
* Separate Wildflower-specific and non-Wildflower-specific portions of `colmap` helper library
* Separate Wildflower-specific and non-Wildflower-specific portions of `smc_kalman` library
* Design and implement better 3D pose smoothing method than simple interpolation
* Consider moving core of reconstruction algorithm to `numpy`
* Consider moving all of pose pair portion of reconstruction algorithm to `networkx`
* Diagnose bottlenecks in reconstruction algorithms
* Set up defaults for visualization functions
* Switch parallel overlay code back to `imap_unordered()` (for less chunky progress bars) but sort output before concatenating
* Ensure that all visual specs (colors, line widths, etc.) propagate to video overlay
* Add drawing primitive to `wf-cv-utils` for text with background
* Use new text-with-background drawing primitive for pose labels
* Add timestamp to video overlays
* Rewrite all log messages so formatting isn't called if log isn't printed
* Make functions handle empty poses (all keypoints `NaN`) more gracefully (e.g., `score_pose_pairs()`, `draw_pose_2d()`)
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
