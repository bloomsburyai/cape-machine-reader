import sys
import os
import subprocess
from package_settings import NAME, VERSION, PACKAGES, DESCRIPTION
from setuptools import setup

# TODO is there a better way ? dependency_links seems to be deprecated and to require a version
_THIS_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if "dependency" in os.path.split(_THIS_FOLDER)[1]:
    pip3_executable = os.path.join(os.path.abspath(os.path.join(os.path.dirname(sys.executable))), 'pip3')
    if not os.path.isfile(pip3_executable):
        pip3_executable = 'pip3'
    command = subprocess.run(['pip3', 'install', '--upgrade', '-r', 'requirements.txt'], check=True)
setup(
    name=NAME,
    version=VERSION,
    long_description=DESCRIPTION,
    author='Bloomsbury AI',
    author_email='contact@bloomsbury.ai',
    packages=PACKAGES,
    include_package_data=True,
    package_data={
        '': ['*.*'],
    },
)
