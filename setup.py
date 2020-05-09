import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

mf = 'microstructure_fingerprinting'

setuptools.setup(
    name=(mf+"-rensonnetg"),
    version="0.1.0",
    author="Gaetan Rensonnet",
    author_email="gaetan.rensonnet@epfl.ch",
    description="Microstructure fingerprinting from DW-MRI data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rensonnetg/microstructure_fingerprinting",
    packages=[mf],  # setuptools.find_packages(),
    # py_modules=["mcf", "mf_utils"], # ??
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.0',  # TODO: check!
    package_dir={mf: '%s' % (mf,)},
    package_data={mf: ['MCF_data/*.mat']},  # MCF matrices data
)
