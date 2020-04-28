# process_pose_data

Tools for fetching, processing, visualizing, and analyzing Wildflower human pose data

## Task list

* Add functions for extracting random timestamp, camera pair
* Add score filter to match identification algorithm (or do we always want to filter pose pairs first?)
* Get pose video overlays working again (for data with track labels)
* Add pipeline stage to combine pose matches across camera view (using `networkx`, as before?)
* Add option of specifying Honeycomb client info for visualization functions that require Honeycomb
* Reinstate `sns.set()` for Seaborn plots without making it spill over into non-Seaborn plots (see [here](https://stackoverflow.com/questions/26899310/python-seaborn-to-reset-back-to-the-matplotlib))
* Refactor code in `visualize` to make it less repetitive (same pattern over and over for `[verb]_by_camera`)
* Fix up legend on pose track timelines
* Add visualization for number of poses per camera per timestamp
* Figure out inconsistent behavior of groupby-apply (under what conditions does it add grouping variables to index?)
* For functions that act on dataframes, make it optional to check dataframe structure (e.g., only one timestamp and camera pair)
* For functions than iterate over previous functions, making naming and approach consistent (e.g., always use apply?)
* For functions that act on dataframes, be consistent about `inplace` option
* Restructure `process_poses_by_timestamp` to use the set of functions above
* Incorporate `tqdm.auto` to get rid of all of the `if` statements for notebook mode
* Implement `tqdm` progress bars for `groupby(...).apply(...)` loops
* Fix up `fetch` module to match design of `wf-video-io`
  - Allow user to supply Honeycomb client
  - Clean up default setting
  - Other?
* Replace `cv.triangulatePoints()` to increase speed (and hopefully accuracy)
* Write function which outputs 3D poses based on selected pose pairs
* Write function which pushes 3D poses to Honeycomb
* Rewrite geom rendering functions to handle the possibility of no track labels
* Rewrite function which overlays geoms on videos so that user can specify a time span that it is a subset of the geoms and/or the video
