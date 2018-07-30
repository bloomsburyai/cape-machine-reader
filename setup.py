import os
import sys
import subprocess
from package_settings import NAME, VERSION, PACKAGES, DESCRIPTION
from setuptools import setup

# TODO is there a better way ? dependencies seem to always require the version
# Calling only at the egg_info step gives us the wanted depth first behavior
if 'egg_info' in sys.argv and os.getenv('CAPE_DEPENDENCIES', 'False').lower() == 'true':
    pip3_executable = os.path.join(os.path.abspath(os.path.join(os.path.dirname(sys.executable))), 'pip3')
    if not os.path.isfile(pip3_executable):
        pip3_executable = 'pip3'
    subprocess.check_call([pip3_executable, 'install', '--upgrade', '-r', 'requirements.txt'])

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
