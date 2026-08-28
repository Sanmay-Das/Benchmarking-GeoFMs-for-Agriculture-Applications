
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(
        _os.path.join(_d, 'configs', 'paths.py')):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _os.path.join(_d, 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS  # noqa: E402

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
        "mmsegmentation @ file://$MSR_ROOT/mmsegmentation",
        "rasterio",
        "rioxarray",
        "einops",
        "timm==0.4.12",
        "tensorboard",
        "imagecodecs",
        "yapf==0.40.1",
    ],
)