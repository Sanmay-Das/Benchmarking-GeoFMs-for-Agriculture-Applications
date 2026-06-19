from setuptools import setup

setup(
    name="geospatial_fm",
    version="0.1.0",
    description="MMSegmentation classes for geospatial-fm finetuning",
    author="Paolo Fraccaro, Carlos Gomes, Johannes Jakubik",
    packages=["geospatial_fm"],
    license="Apache 2",
    long_description=open("README.md").read(),
    install_requires=[
        "mmsegmentation @ file:///bigdata/eldawylab/sdas050/MS_Research/mmsegmentation",
        "rasterio",
        "rioxarray",
        "einops",
        "timm==0.4.12",
        "tensorboard",
        "imagecodecs",
        "yapf==0.40.1",
    ],
)