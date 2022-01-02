from setuptools import setup, find_packages
import re
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))

VERSION_RE = re.compile(r'''__version__ = ['"]([0-9.]+)['"]''')

def get_version():
    init = open(os.path.join(BASEDIR, 'poseconnect', '__init__.py')).read()
    return VERSION_RE.search(init).group(1)

# Dependencies (format is 'PYPI_PACKAGE_NAME[>]=VERSION_NUMBER')
BASE_DEPENDENCIES = [
    'wf-smc-kalman>=0.1.0',
    'wf-cv-utils>=3.3.0',
    'pandas>=1.2.2',
    'numpy>=1.20.1',
    'networkx>=2.5',
    # 'opencv-python>=4.5.1',
    'matplotlib>=3.3.4',
    'seaborn>=0.11.1',
    'tqdm>=4.57.0',
    'python-slugify>=4.0.1',
    'python-dateutil>=2.8',
    'click>=8.0.0'
]

# TEST_DEPENDENCIES = [
# ]
#
# LOCAL_DEPENDENCIES = [
# ]

# Allow setup.py to be run from any path
os.chdir(os.path.normpath(BASEDIR))

setup(
    name='poseconnect',
    packages=find_packages(),
    version=get_version(),
    include_package_data=True,
    description='Tools for constructing 3D pose tracks from multi-camera 2D poses',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/WildflowerSchools/poseconnect',
    author='Theodore Quinn',
    author_email='ted.quinn@wildflowerschools.org',
    install_requires=BASE_DEPENDENCIES,
    # tests_require=TEST_DEPENDENCIES,
    # extras_require = {
    #     'test': TEST_DEPENDENCIES,
    #     'local': LOCAL_DEPENDENCIES
    # },
    entry_points={
        "console_scripts": [
             "poseconnect = poseconnect.cli:cli"
        ]
    },
    keywords=['pose reconstruction', 'pose tracking'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
