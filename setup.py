from setuptools import setup, find_packages

VERSION = "1.0.0" 
DESCRIPTION = """
A Python library called ScratchML was created to build the most fundamental Machine Learning models from scratch, emphasizing producing user-friendly, straightforward, and easy-to-use implementations for novices and enthusiasts.
"""

setup(
    name="ScratchML", 
    version=VERSION,
    author="Rafael Greca Vieira",
    author_email="rgvieira97@gmail.com",
    description=DESCRIPTION,
    packages=find_packages(),
    keywords=[
        "python",
        "scratch",
        "machine learning",
        "linear regression",
        "logistic regression"
    ]
)