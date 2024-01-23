[![name](https://colab.research.google.com/assets/colab-badge.svg)]([[https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/ChemNLP_Example.ipynb]](https://colab.research.google.com/drive/1FRayOxp07ReOUUrL7ZXkPTmF1Ocu5ygI?usp=sharing))
![alt text](https://github.com/usnistgov/chemnlp/actions/workflows/main.yml/badge.svg)
[![DOI](https://zenodo.org/badge/523320947.svg)](https://zenodo.org/badge/latestdoi/523320947)

# Chemistry-NLP

# New Releases
* Text classification using different algorithms
* Feature selection
* Dimesionality reduction
* Clustering
* Words Prediction
  
# Table of Contents
* [Introduction](#intro)
* [Installation](#install)
* [Examples](#example)
* [Web-app](#webapp)
* [Reference](#reference)

<a name="intro"></a>
Introduction
-------------------------
Chemistry-NLP is a software-package to process chemical information from the scientific literature.

<p align="center">
   <img src="https://github.com/zaki1003/Chemistry-NLP/blob/develop/chemnlp/Schemcatic.PNG" alt="ChemNLP"  width="600"/>
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
Now, let's make a conda environment, say "Chemistry-NLP", choose other name as you like::
```
conda create --name Chemistry-NLP python=3.9
source activate Chemistry-NLP
```
#### Method 1 (using setup.py):

Now, let's install the package:
```
git clone https://github.com/zaki1003/Chemistry-NLP.git
cd Chemistry-NLP
python setup.py develop
cde data download
```

#### Method 2 (using pypi):

As an alternate method, Chemistry-NLP can also be installed using `pip` command as follows:
```
pip install chemnlp
cde data download
```


<a name="classification"></a>
Classification
---------
For classification I tried 7 classification algorithms: SVC, MLPClassifier, RandomForestClassifier, Logistic, XGBoost, KNN, MultinomialNB. 

### Feature selection in text classification:
Feature selection is one of the most important steps in the field of text classification. As text data mostly have high dimensionality problems. To reduce the curse of high dimensionality, feature selection techniques are used. The basic idea behind feature selection is to keep only important features and remove less contributing features.

Issues associated with high dimensionality are as follows:

1. Adds unnecessary noise to the model

2. High space and time complexity

3. Overfitting


The feature selection techniques we used in our project are: 

the Chi-Square feature selection: The Chi-square test is used in statistics to test the independence of two events. More specifically in feature selection, we use it to test whether the occurrence of a specific term and the occurrence of a specific class are independent.

f_classif : Compute the ANOVA F-value between label/feature for classification tasks.

mutual_info_classif : Estimate mutual information for a discrete target variable

### Dimensionality reduction in text classification:
In the field of Natural Language Processing (NLP), analyzing and processing vast amounts of text data can be challenging. Dimensionality reduction techniques come to our rescue by simplifying the data and extracting meaningful information.

The dimesionality reduction technique we used in our project is: 
TruncatedSVD: This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. This means it can work with sparse matrices efficiently. And it works efficiently on count/tf-idf vactors.

**NB**: The PCA didn't work because it does not support sparse input.

### Classification Results:
<table>
    <thead>
        <tr>
            <th rowspan=2></th>
            <th rowspan=2>original data</th>
            <th colspan=3>Feature Selection with k_best=1500</th>
            <th>Dimonsionality Reduction with n_component=20</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td ></td>
            <td></td>
            <td>chi-square</td>
            <td>  mutual_info_classif</td>
            <td>f_classif</td>
            <td>Trunced SVD</td>
        </tr>
        <tr>
            <td >SVC</td>
            <td><b>0.94</b></td>
            <td> 0.90</td>
            <td> 0.90 </td>
            <td>0.90 </td>
            <td> 0.81 </td>
        </tr>
        <tr>
            <td >MLPClassifier</td>
            <td> <b>0.94</b></td>
            <td> 0.87 </td>
            <td>  0.87 </td>
            <td> 0.866 </td>
            <td> 0.84</td>
        </tr>
        <tr>
            <td >RandomForestClassifier</td>
            <td><b>0.94</b>  </td>
            <td> <b> 0.93 </b></td>
            <td>   <b> 0.94 </b> </td>
            <td> <b>0.93</b> </td>
            <td> 0.93  </td>
        </tr>
        <tr>
            <td >Logistic regression </td>
            <td> <b>0.92</b></td>
            <td>0.88</td>
            <td> 0.89 </td>
            <td> 0.88 </td>
            <td> 0.80 </td>
        </tr>
        <tr>
            <td >XGBoost</td>
            <td> <b>0.91</b></td>
            <td> <b>0.90</b> </td>
            <td> <b>0.90</b> </td>
            <td><b>0.90</b></td>
            <td>0.93</td>
        </tr>
        <tr>
            <td >KNN</td>
            <td> 0.53</td>
            <td> 0.84 </td>
            <td> 0.82 </td>
            <td>0.84</td>
            <td><b>0.91 (here KNN is the fastest)</b></td>
        </tr>
        <tr>
            <td >MultinomialNB</td>
            <td> <b>0.89</b></td>
            <td> 0.85 </td>
            <td> 0.86 </td>
            <td>0.85</td>
            <td> 0.53 (doesn't support negative values)</td>
        </tr>
    </tbody>
</table>

### Text classification example
#### SVC
```
!python chemnlp/classification/classification.py --csv_path pubchem.csv --key_column label_name --value_column title --value_column title --classifiction_algorithm SVC
```
#### Feature Selection + RandomForestClassifier
```
!python chemnlp/classification/classification.py --csv_path pubchem.csv --key_column label_name --value_column title --value_column title --classifiction_algorithm RandomForestClassifier --do_feature_selection True --feature_selection_algorithm chi2
```
#### Dimonsionality Reduction + KNN
```
!python chemnlp/classification/classification.py --csv_path pubchem.csv --key_column label_name --value_column title --value_column title --classifiction_algorithm KNN --do_dimonsionality_reduction True --n_components 20
```


<a name="clustering"></a>
Clustering
---------
#### abstract clustering

```
!python chemnlp/clustering/clustering.py --clustering_algorithm KMeans 
```
#### result:
![Capture d'écran 2024-01-22 185442](https://github.com/usnistgov/chemnlp/assets/65148928/c4564941-cccd-4157-83f0-43620b93ff29)





<a name="example"></a>
Examples
---------
#### Parse chemical formula 

```
run_chemnlp.py --file_path="chemnlp/tests/XYZ"
```



[Google Colab example](https://github.com/zaki1003/Chemistry-NLP/blob/main/Chemistry_NLP_(MERABET).ipynb)



<a name="webapp"></a>
Using the webapp
---------
The webapp is available at: https://jarvis.nist.gov/jarvischemnlp

![JARVIS-ChemNLP](https://github.com/usnistgov/chemnlp/blob/develop/chemnlp/PTable.PNG)

<a name="reference"></a>
Reference
---------


[ChemNLP: A Natural Language Processing based Library for Materials Chemistry Text Data](https://arxiv.org/abs/2209.08203)
