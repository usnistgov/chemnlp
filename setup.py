import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chemnlp",
    version="2023.5.5",
    author="Kamal Choudhary",
    author_email="kamal.choudhary@nist.gov",
    description="chemnlp",
    install_requires=[
<<<<<<< HEAD
        "numpy>=1.22.0",
        "scipy>=1.6.3",
        "jarvis-tools>=2021.07.19",
        "ChemDataExtractor>=1.3.0",
        "matplotlib>=3.4.1",
        "flake8>=3.9.1",
        "pycodestyle>=2.7.0",
        "pydocstyle>=6.0.0",
=======
       "numpy>=1.22.0",
       "scipy>=1.6.3",
       "jarvis-tools>=2021.07.19",
       "ChemDataExtractor>=1.3.0",
       "matplotlib>=3.4.1",
       "flake8>=3.9.1",
       "pycodestyle>=2.7.0",
       "pydocstyle>=6.0.0",
>>>>>>> 54b236a4b5d43c6c2fb1a8877b9d380e1dceeda0
    ],
    scripts=["chemnlp/utils/run_chemnlp.py"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usnistgov/chemnlp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
