# process_pose_data

Tools for fetching, processing, visualizing, and analyzing Wildflower human pose data

## Task list
* Fix problem in heat map for empty data
* Add outer frame on heat map
* Add keypoint connectors to pose drawing
* Get control of color cycle for pose drawing
* Add functions for extracting random timestamp, camera pair
* Add function which produces heatmap and both camera views for chosen timestamp, camera pair
* Rewrite geom rendering functions to handle the possibility of no track labels
* Rewrite function which overlays geoms on videos so that user can specify a time span that it is a subset of the geoms and/or the video
* Write function(s) which select(s) pose pairs based on score
* Write function which outputs 3D poses based on selected pose pairs
* Write function which pushes 3D poses to Honeycomb
