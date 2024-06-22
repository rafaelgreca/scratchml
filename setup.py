from setuptools import setup, find_packages

VERSION = "3.0.0"
DESCRIPTION = """
A Python library called ScratchML was created to build the most fundamental Machine Learning models from scratch, emphasizing producing user-friendly, straightforward, and easy-to-use implementations for novices and enthusiasts.
"""

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements/requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read()

with open("requirements/requirements_test.txt", "r", encoding="utf-8") as f:
    test_requires = f.read()

setup(
    name="ScratchML",
    version=VERSION,
    author="Rafael Greca Vieira",
    author_email="rgvieira97@gmail.com",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["scratchml"]),
    install_requires=install_requires,
    tests_require=test_requires,
    test_suite="tests",
    keywords=[
        "python",
        "scratch",
        "machine learning",
        "linear regression",
        "logistic regression",
    ],
    python_requires=">=3.11",
)
