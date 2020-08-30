import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = open(os.path.join(BASEDIR, 'VERSION')).read().strip()

# Dependencies (format is 'PYPI_PACKAGE_NAME[>=]=VERSION_NUMBER')
BASE_DEPENDENCIES = [
    'wf-pose-tracking-3d>=0.1.0',
    'wf-smc-kalman>-0.1.0',
    'wf-cv-utils>=0.5.1',
    'wf-video-io>=0.1.0',
    'ffmpeg-python>=0.2.0',
    'wf-minimal-honeycomb-python>=0.5.0',
    'wf-geom-render>=0.3.0',
    'pandas>=0.25.3',
    'numpy>=1.18.1',
    'networkx>=2.4',
    'tqdm>=4.42.0',
    'opencv-python>=4.2.0.34',
    'python-slugify>=4.0.0',
    'matplotlib>=3.1.2',
    'seaborn>=0.10.0'
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
    keywords=['pose estimation'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
