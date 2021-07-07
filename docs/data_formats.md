# Data formats

Internally, the main data objects (2D pose data, camera calibration info, 3D pose data with or without track IDs and/or person IDs) are represented by [pandas](https://pandas.pydata.org/) `DataFrame` objects.

For file I/O, the library supports a number of different formats:

* JSON (`.json` extension): A text file in JSON format with data structured as either:

  * A `list` of `dict` objects, where the keys of each dict are the field names (including the index field), or

  * A `dict`, where the keys are index values and the values are `dict` objects with the non-index fields as keys

* Pickled `DataFrame` (`.pkl` and `.pickle` extensions): A `pandas` `DataFrame` object serialized to `pickle` format using either `DataFrame.to_pickle(...)` or the native Python `pickle.dump(...)`

* (Input only) Comma-separated value (`.csv` extension): A text file with a header containing the field names as column names

Details of the required fields for each type of data are given below (other fields will be ignored).

ID fields are in `UUID` format when generated internally but any hashable text values should work.

Timestamps should be timezone-aware. ISO format is preferred but library will accept any format recognized by `pandas.to_datetime(...)`. Currently, the library only supports integer frame rates with frames falling on even millisecond boundaries (e.g., 12:30:00.000, 12:30:00.100, 12:30:00.200, etc.).

Array and vector values are represented internally (and in `pickle` files) by [numpy](https://numpy.org/) arrays. In JSON and CSV files, they should be encoded as lists (or lists of lists). CSV files require that the list be JSON encoded and quoted.

### 2D pose data

* `pose_2d_id` (index field): Text field with a unique ID for each 2D pose
* `timestamp`: Timezone-aware timestamp for each 2D pose
* `camera_id`: Text field with a the ID of the camera that captured the pose
* `keypoint_coordinates_2d`: An Nx2 array of floats with (where N is the number of keypoints in the chosen pose model) with the image coordinates of each keypoint
* `keypoint_quality_2d`: An vector of length N (where N is the number of keypoints in the chosen pose model) with the quality score of each keypoint in the pose
* `pose_quality_2d`: A float representing the overall quality score of the pose

### Camera calibration info

See the [OpenCV docs](https://docs.opencv.org/4.5.2/d9/d0c/group__calib3d.html) for details of the calibration values.

* `camera_id` (index field): Text field with a unique ID for each camera
* `camera_matrix`: A 3x3 array of floats with principal point and focal length info for the camera
* `distortion_coefficients`: A vector of floats (length depends on distortion model) representing with information about the lens distortion of the camera
* `rotation_vector`: A vector of 3 floats representing the orientation of the camera with respect to the world coordinates
* `translation_vector`: A vector of 3 floats representing the position of the camera with respect to the world coordinates
* `image_width`: Integer representing the width of the camera image in pixels
* `image_height`: Integer representing the height of the camera image in pixels

### 3D pose data

* `pose_3d_id` (index field): Text field with a unique ID for each 3D pose
* `timestamp`: Timezone-aware timestamp for each 3D pose
* `keypoint_coordinates_3d`: An Nx3 array of floats with (where N is the number of keypoints in the chosen pose model) with the world coordinates of each keypoint in the pose
* `pose_2d_ids`: A list of 2D pose IDs for the 2D poses from which this 3D pose  was constructed (this field will be empty if pose resulted from interpolation)
* `pose_track_3d_id` (where relevant): Text field with unique ID for each pose track
* `person_id` (where relevant): Text field with the ID of the person associated with this pose track (this field will be empty if algorithm fails to associate a person)

### Location sensor data

* `position_id` (index field): Text field with a unique ID for each sensor reading
* `timestamp`: Timezone-aware timestamp for each sensor reading
* `person_id`: Text field with the ID of the person associated with the sensor reading (typically the person wearing the sensor)
* `x_position`: Float representing the _x_ coordinate of the sensor reading
* `y_position`: Float representing the _y_ coordinate of the sensor reading
* `z_position`: Float representing the _z_ coordinate of the sensor reading
