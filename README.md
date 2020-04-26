# process_pose_data

Tools for fetching, processing, visualizing, and analyzing Wildflower human pose data

## Task list
* For functions that act on dataframes, make it optional to check dataframe structure (e.g., only one timestamp and camera pair)
* For functions than iterate over previous functions, making naming and approach consistent (e.g., always use apply?)
* Fix up interaction between `generate_pose_pairs()` and `generate_pose_pairs_timestamp()` (why is it messing with index?)
* For functions that act on dataframes, be consistent about `inplace` option
* Restructure `process_poses_by_timestamp` to use the two set of functions above
* Add option of specifying Honeycomb client info for visualization functions that require Honeycomb
* Reinstate `sns.set()` for Seaborn plots without making it spill over into non-Seaborn plots (see [here](https://stackoverflow.com/questions/26899310/python-seaborn-to-reset-back-to-the-matplotlib))
* Refactor code in `visualize` to make it less repetitive (same pattern over and over for `[verb]_by_camera`)
* Fix up legend on pose track timelines
* Add visualization for number of poses per camera per timestamp
* Add functions for extracting random timestamp, camera pair
* Add function which produces heatmap and both camera views for chosen timestamp, camera pair
* Fix up `fetch` module to match design of `wf-video-io`
  - Allow user to supply Honeycomb client
  - Clean up default setting
  - Other?
* Replace `cv.triangulatePoints()` to increase speed (and hopefully accuracy)
* Write function(s) which select(s) pose pairs based on score
* Write function which outputs 3D poses based on selected pose pairs
* Write function which pushes 3D poses to Honeycomb
* Rewrite geom rendering functions to handle the possibility of no track labels
* Rewrite function which overlays geoms on videos so that user can specify a time span that it is a subset of the geoms and/or the video
