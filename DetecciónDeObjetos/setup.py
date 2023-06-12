from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Una herramienta de detección de elementos peculiares'
LONG_DESCRIPTION = 'Falta'

# Setting up
setup(
    name="co2ncientes",
    version=VERSION,
    author="Co2ncientes (CODEFEST 2023)",
    author_email="ma.acostaw@uniandes.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "easyocr",
        "numpy",
        "object_detection",
        "opencv_python",
        "opencv_python_headless",
        "tensorflow",
        "tensorflow_intel",
    ],
    keywords=['python', 'CODEFEST'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)