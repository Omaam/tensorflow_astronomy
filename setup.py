"""Setup.
"""
from setuptools import find_packages
from setuptools import setup

setup(
    name="tensorflow_astronomy",
    version="0.0.1",
    description=(
        "Astronomical data processing and analysis tools in TensorFlow"
    ),
    author="Tomoki Omama",
    packages=find_packages(),
    install_requires=[
        "astropy",
        "numpy",
        "tensorflow",
        "tensorflow_probability"
    ],
    classifiers=[
        "Development Status :: 1 - Planning"
    ],
)
