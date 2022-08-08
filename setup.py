from setuptools import setup, find_packages

setup(
    name="looming_spots",
    version="",
    packages=find_packages(),
    url="https://github.com/SainsburyWellcomeCentre/looming_spots.git",
    license="MIT",
    author="Stephen Lenzi",
    author_email="s.lenzi@ucl.ac.uk",
    description="Analysis of loom-evoked escape experiments in python",
    install_requires=[
        "numpy",
        "configobj",
        "matplotlib",
        "seaborn",
        "scipy",
        "pandas",
        "scikit-video",
        "pingouin",
        "nptdms",
        "cached-property",
        "tqdm",
        "scikit-image",
        "tables",
        "more_itertools",
        "pims"
    ],
)
