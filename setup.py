from setuptools import setup, find_packages

setup(
    name="looming_spots",
    version="",
    packages=find_packages(exclude=["tests", "doc", "deprecated"]),
    url="https://github.com/stephenlenzi/looming_spots.git",
    license="MIT",
    author="Stephen Lenzi",
    author_email="s.lenzi@ucl.ac.uk",
    description="looming spot analysis in python",
    install_requires=[
        "numpy",
        "configobj",
        "matplotlib",
        "seaborn",
        "scipy",
        "pandas",
        "pims",
        "scikit-video",
        "pingouin",
    ],
)
