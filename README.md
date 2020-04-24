# process_pose_data

Tools for fetching, processing, visualizing, and analyzing Wildflower human pose data

## Task list
* Add option of specifying Honeycomb client info for visualization functions that require Honeycomb
* Get control of color cycle for pose drawing
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
