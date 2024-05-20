from setuptools import setup, find_packages

VERSION = "0.0.1" 
DESCRIPTION = "My first Python package"
LONG_DESCRIPTION = "My first Python package with a slightly longer description"

setup(
    name="scratchml", 
    version=VERSION,
    author="Rafael Greca Vieira",
    author_email="rgvieira97@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    keywords=["python", "scratch", "machine learning"]
)