import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = open(os.path.join(BASEDIR, 'VERSION')).read().strip()

# Dependencies (format is 'PYPI_PACKAGE_NAME[>=]=VERSION_NUMBER')
BASE_DEPENDENCIES = [
    'wf-pose-tracking-3d>=0.1.0',
    'wf-smc-kalman>-0.1.0',
    'wf-cv-utils>=0.5.4',
    'wf-video-io>=0.1.0',
    'ffmpeg-python>=0.2.0',
    'wf-minimal-honeycomb-python>=0.6.0',
    'wf-geom-render>=0.3.0',
    'pandas>=1.0.5',
    'numpy>=1.19.0',
    'networkx>=2.4',
    'tqdm>=4.47.0',
    'opencv-python>=4.3.0.36',
    'python-slugify>=4.0.1',
    'matplotlib>=3.2.2',
    'seaborn>=0.10.1',
    'click>=7.1.2'
]

# TEST_DEPENDENCIES = [
# ]
#
# LOCAL_DEPENDENCIES = [
# ]

# Allow setup.py to be run from any path
os.chdir(os.path.normpath(BASEDIR))

setup(
    name='wf-process-pose-data',
    packages=find_packages(),
    version=VERSION,
    include_package_data=True,
    description='Tools for fetching, processing, visualizing, and analyzing Wildflower human pose data',
    long_description=open('README.md').read(),
    url='https://github.com/WildflowerSchools/wf-process-pose-data',
    author='Theodore Quinn',
    author_email='ted.quinn@wildflowerschools.org',
    install_requires=BASE_DEPENDENCIES,
    # tests_require=TEST_DEPENDENCIES,
    # extras_require = {
    #     'test': TEST_DEPENDENCIES,
    #     'local': LOCAL_DEPENDENCIES
    # },
    entry_points='''
        [console_scripts]
        reconstruct_poses_3d=process_pose_data.workers:reconstruct_poses_3d_alphapose_local_by_time_segment
    ''',
    keywords=['pose estimation'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
