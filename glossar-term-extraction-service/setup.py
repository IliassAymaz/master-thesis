# coding: utf-8

import sys
from setuptools import setup, find_packages

NAME = "GTETE_backend"
VERSION = "1.0.0"

# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

REQUIRES = ["connexion"]

setup(
    name=NAME,
    version=VERSION,
    description="Glossar Term Extraction",
    author_email="hussein.hasso@fkie.fraunhofer.de",
    url="",
    keywords=["Swagger", "Glossar Term Extraction"],
    install_requires=REQUIRES,
    packages=find_packages(),
    package_data={'': ['swagger/swagger.yaml']},
    include_package_data=True,
    entry_points={
        'console_scripts': ['GTETE_backend=GTETE_backend.__main__:main']},
    long_description="""\
    Microservice for the extraction of glossar terms in software requirements.
    """
)

