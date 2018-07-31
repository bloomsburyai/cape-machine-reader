from package_settings import NAME, VERSION, PACKAGES, DESCRIPTION
from setuptools import setup

setup(
    name=NAME,
    version=VERSION,
    long_description=DESCRIPTION,
    author='Bloomsbury AI',
    author_email='contact@bloomsbury.ai',
    packages=PACKAGES,
    include_package_data=True,
    install_requires=[
        'pytest==3.6.4',
        'numpy==1.15.0',
        'pyinterval==1.2.0',
    ],
    package_data={
        '': ['*.*'],
    },
)
