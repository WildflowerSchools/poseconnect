import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = open(os.path.join(BASEDIR, 'VERSION')).read().strip()

# Dependencies (format is 'PYPI_PACKAGE_NAME[>=]=VERSION_NUMBER')
# BASE_DEPENDENCIES = [
# ]

# TEST_DEPENDENCIES = [
# ]
#
# LOCAL_DEPENDENCIES = [
# ]

# Allow setup.py to be run from any path
os.chdir(os.path.normpath(BASEDIR))

setup(
    name='PYPI_PACKAGE_NAME',
    packages=find_packages(),
    version=VERSION,
    include_package_data=True,
    description='SHORT_DESCRIPTION',
    long_description=open('README.md').read(),
    url='https://github.com/WildflowerSchools/PYPI_PACKAGE_NAME',
    author='AUTHOR_NAME',
    author_email='AUTHOR_EMAIL',
    # install_requires=BASE_DEPENDENCIES,
    # tests_require=TEST_DEPENDENCIES,
    # extras_require = {
    #     'test': TEST_DEPENDENCIES,
    #     'local': LOCAL_DEPENDENCIES
    # },
    # keywords=['KEYWORD'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
