[![name](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ChemNLP_Example.ipynb](https://colab.research.google.com/drive/1FRayOxp07ReOUUrL7ZXkPTmF1Ocu5ygI?usp=sharing))
![alt text](https://github.com/usnistgov/chemnlp/actions/workflows/main.yml/badge.svg)
[![DOI](https://zenodo.org/badge/523320947.svg)](https://zenodo.org/badge/latestdoi/523320947)


# ChemNLP

# Table of Contents
* [Introduction](#intro)
* [Installation](#install)
* [Examples](#example)
* [Web-app](#webapp)
* [Reference](#reference)

<a name="intro"></a>
Introduction
-------------------------
ChemNLP is a software-package to process chemical information from the scientific literature.

<p align="center">
   <img src="https://github.com/usnistgov/chemnlp/blob/develop/chemnlp/Schemcatic.PNG" alt="ChemNLP"  width="600"/>
</p>

<a name="install"></a>
Installation
-------------------------
First create a conda environment:
Install miniconda environment from https://conda.io/miniconda.html
Based on your system requirements, you'll get a file something like 'Miniconda3-latest-XYZ'.

Now,

```
bash Miniconda3-latest-Linux-x86_64.sh (for linux)
bash Miniconda3-latest-MacOSX-x86_64.sh (for Mac)
```
Download 32/64 bit python 3.8 miniconda exe and install (for windows)
Now, let's make a conda environment, say "chemnlp", choose other name as you like::
```
conda create --name chemnlp python=3.9
source activate chemnlp
```
#### Method 1 (using setup.py):

Now, let's install the package:
```
git clone https://github.com/usnistgov/chemnlp.git
cd chemnlp
python setup.py develop
cde data download
```

#### Method 2 (using pypi):

As an alternate method, ChemNLP can also be installed using `pip` command as follows:
```
pip install chemnlp
cde data download
```

<a name="example"></a>
Examples
---------
#### Parse chemical formula 

```
run_chemnlp.py --file_path="chemnlp/tests/XYZ"
```

#### Text classification example

```
python chemnlp/classification/scikit_class.py --csv_path chemnlp/sample_data/cond_mat_small.csv
```

[Google Colab example for installation and text classification](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ChemNLP_Example.ipynb)

#### Text generation example

[Google Colab example for Text Generation with HuggingFace](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ChemNLP_TitleToAbstract.ipynb)


<a name="webapp"></a>
Using the webapp
---------
The webapp is available at: https://jarvis.nist.gov/jarvischemnlp

![JARVIS-ChemNLP](https://github.com/usnistgov/chemnlp/blob/develop/chemnlp/PTable.PNG)

<a name="reference"></a>
Reference
---------


[ChemNLP: A Natural Language Processing based Library for Materials Chemistry Text Data](https://arxiv.org/abs/2209.08203)
