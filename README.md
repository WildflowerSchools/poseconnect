# process_pose_data

Tools for fetching, processing, visualizing, and analyzing Wildflower human pose data

## Task list
* Implement more flexible function for fetching 2D pose data
  - Search on properties of inference execution, pose model, etc.
  - Control what gets returned (e.g., track labels, person assignments, pose model IDs, etc.)
* Rewrite existing functions for fetching 2D pose data as wrappers for the above
* Rewrite geom rendering functions to handle the possibility of no track labels
* Write function to download and concatenate videos
* Rewrite function which overlays geoms on videos so that user can specify a time span that it is a subset of the geoms and/or the video
* Streamline and test function that produces potential pose pairs across cameras
* Write function(s) which score(s) pose pairs
* Write function(s) which select(s) pose pairs based on score
* Write function which outputs 3D poses based on selected pose pairs
* Write function which pushes 3D poses to Honeycomb
