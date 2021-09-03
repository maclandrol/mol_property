# from distutils.core import setup
from setuptools import setup
from setuptools import find_packages

setup(
    name="mol_property",
    version="1.0.0",
    description="Prediction of pKa from chemical structure using machine learning approaches",
    long_description=open("README.md", encoding="utf-8").read(),
    author="Chin",
    author_email="chin340823@163.com",
    url="https://github.com/TVect/mol_property",
    packages=find_packages(),
    package_data={
        "mol_property": [
            "mol_property/pka/model/*.pkl",
            "mol_property/similarity/save/*.zip",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    zip_safe=False,
)
