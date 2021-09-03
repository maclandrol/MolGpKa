from setuptools import setup
from setuptools import find_packages
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

setup(
    name="molgpka",
    version="1.0.0",
    author="xiaolinpan",
    description="Prediction of pka for molecules.",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    entry_points={
        "console_scripts": [
            "molgpka=molgpka.cli:main_cli",
        ],
    },
)
