from setuptools import setup, find_packages
import re
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))

VERSION_RE = re.compile(r'''__version__ = ['"]([0-9.]+)['"]''')

def get_version():
    init = open(os.path.join(BASEDIR, 'process_pose_data', '__init__.py')).read()
    return VERSION_RE.search(init).group(1)

# Dependencies (format is 'PYPI_PACKAGE_NAME[>]=VERSION_NUMBER')
BASE_DEPENDENCIES = [
    'wf-honeycomb-io>=0.0.1',
    'wf-smc-kalman>=0.1.0',
    'wf-cv-utils>=3.0.0',
    'wf-video-io>=1.0.0',
    'wf-geom-render>=0.3.0',
    'pandas>=1.2.2',
    'numpy>=1.20.1',
    'networkx>=2.5',
    'opencv-python>=4.5.1',
    'matplotlib>=3.3.4',
    'seaborn>=0.11.1',
    'ffmpeg-python>=0.2.0',
    'tqdm>=4.57.0',
    'python-slugify>=4.0.1',
    'python-dateutil>=2.8'
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
    version=get_version(),
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
    # entry_points={
    #     "console_scripts": [
    #          "COMMAND_NAME = MODULE_PATH:METHOD_NAME"
    #     ]
    # },
    keywords=['pose estimation'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
