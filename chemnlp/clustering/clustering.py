"""Module for classification tasks."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split
import argparse
import xgboost as xgb
from sklearn.metrics.cluster import normalized_mutual_info_score
import sys
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from time import time
import nltk
from mclustpy import mclustpy
import numpy as np
from mclustpy import mclustpy
# from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from jarvis.db.figshare import data
from jarvis.db.jsonutils import dumpjson
import pickle
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="ChemNLP package.")
parser.add_argument(
    "--csv_path",
    default="filename.csv",
    help="Path for comma separated file.",
)

parser.add_argument(
    "--classifiction_algorithm",
    default="SVC",

)
parser.add_argument(
    "--do_feature_selection",
    default=False,
)
parser.add_argument(
    "--feature_selection_algorithm",
    default="chi2",
)
parser.add_argument(
    "--do_dimonsionality_reduction",
    default=False,
)
parser.add_argument(
    "--k_best",
    default=1500,
)
parser.add_argument(
    "--n_components",
    default=20,
)




parser.add_argument(
    "--test_ratio",
    default=0.2,
    help="Test split ratio e.g. 0.2",
)

parser.add_argument(
    "--key_column",
    default="categories",
    help="Column name for classes in csv file",
)

parser.add_argument(
    "--value_column",
    default="title_abstract",
    help="Column name for text in csv file",
)

parser.add_argument(
    "--min_df",
    default=5,
    help="Minimum numbers of documents a word must be present in to be kept",
)
parser.add_argument(
    "--shuffle",
    default=False,
    help="Whether or not shuffle the rows in csv before splitting",
)



def abstract2df(df=None, idx=None, text=""):
    ps = PorterStemmer()
    """
    USAGE: Use Porter Stemming to process a paper abstract.

    ARGUMENT:
    df - DataFrame with keys ["term", "title", "summary", "id"]
    idx - index locating paper within DataFrame

    RETURNS: stem_df - DataFrame with keys ["stems", "counts", "docs"]
    """
    abstract = df.iloc[idx][text]
    # abstract = df.iloc[idx].summary
    words = word_tokenize(abstract)
    stems = [ps.stem(w).lower() for w in words]
    counts = Counter(stems)
    docs = list(np.ones(len(counts)).astype(int))
    stem_df = pd.DataFrame(
        {
            "stems": list(counts.keys()),
            "counts": list(counts.values()),
            "docs": docs,
        }
    )
    stem_df = stem_df[["stems", "counts", "docs"]]
    return stem_df


def docs2idf(df, num):
    """
    USAGE:
    Get an inverse document frequency (idf) measure for each stem,
    and add that measure to the word stem DataFrame.

    ARGUMENTS:
    df - word stem DataFrame
    num - total number of papers in document pool
    """
    df["idf"] = np.log(num / df["docs"])
    # print ('df',df)
    return df


def reduce_dim(matrix, dims):
    """
    USAGE: Perform Truncated SVD to reduce dimensionality of embedding space.

    ARGUMENTS:
    matrix - training data (sparse matrix) with shape (n_features, n_samples)
    dims - number of dimensions to project data into

    RETURNS:
    X_new - array of reduced dimensionality
    """

    tic = time()
    print("Performing Truncated SVD...")
    svd = TruncatedSVD(n_components=dims, random_state=1)
    X = svd.fit_transform(matrix)  # matrix.shape = (n_samples, n_features)
    toc = time()
    print(
        "Embeddings reduced from %d to %d using TruncatedSVD. (Time: %.2f s)"
        % (matrix.shape[1], dims, (toc - tic))
    )
    return X


def TSNE2D(X):
    """
    USAGE: Perform TSNE to reduce embedding space to 2D
    ARGUMENT: X - high-dimensional training array (n_samples, n_features ~ 100)
    RETURNS: X_embedded - 2D matrix (n_samples, 2)
    """

    tic = time()
    print("Performing TSNE...")
    X_embedded = TSNE(
        n_components=2, perplexity=30, random_state=1
    ).fit_transform(X)
    toc = time()
    print(
        "Embeddings reduced to 2 dimensions through TSNE. (Time: %.2f s)"
        % (toc - tic)
    )
    return X_embedded


def clustering(df=None, category_key="categories", text="title", filename=None, ndim=128):
    print("Counting word stems in %d abstracts..." % len(df))
    samples = len(df)
    full_stem_df = pd.DataFrame()
    for idx in range(len(df)):
        # print(idx)
        stem_df = abstract2df(df=df, idx=idx, text=text)
        full_stem_df = pd.concat([full_stem_df, stem_df])
        if idx and idx % 10 == 0 or idx == len(df) - 1:
            full_stem_df = full_stem_df.groupby("stems").sum()
            full_stem_df["stems"] = full_stem_df.index
            full_stem_df = full_stem_df[["stems", "counts", "docs"]]
    full_stem_df.sort_values("counts", ascending=False, inplace=True)
    all_stems = full_stem_df  # count_stems(dfb)
    all_stems = docs2idf(all_stems, len(df))
    all_stems.reset_index(drop=True, inplace=True)
    # df=dfb
    stems = all_stems
    # assert samples <= len(df)
    idf = np.array(list(stems["idf"]))
    stems = stems[["stems", "idf"]]
    matrix = np.zeros((samples, len(stems)))
    

    info = []
    print("Embedding paper abstracts...")
    i = 0
    for idx in tqdm(range(samples)):
        # print(idx)
        information = list(df.iloc[idx][["id", category_key, text]])
        info.append(information)
        new_df = pd.merge(
            stems,
            abstract2df(df=df, idx=idx, text=text),
            on="stems",
            how="left",
        ).fillna(0)
        vec = np.array(list(new_df["counts"]))
        vec /= np.sum(vec)  # components sum to 1
        vec *= idf  # apply inverse document frequency
        matrix[i, :] = vec
        i += 1
    print("Paper abstracts converted to vectors.")

    # label encoding
    label_encoder = LabelEncoder()
    label_encoder.fit(df[category_key])
    encoded_labels = label_encoder.transform(df[category_key])

    matrix = np.nan_to_num(matrix, nan=0.0)
    print('matrix',matrix )
    X = reduce_dim(matrix, ndim)
    print('X',X )

    X_embedded = TSNE2D(X)
    print('X_embedded',X_embedded )


    kmeans = KMeans(n_clusters=7, random_state=0, n_init="auto").fit(X_embedded)
    kmeans.labels_
    print(kmeans.labels_)
    print(encoded_labels)
    
    print("NMI:", normalized_mutual_info_score(kmeans.labels_,encoded_labels))
    print("ARI:", adjusted_rand_score(kmeans.labels_,encoded_labels))
    
    clustering = DBSCAN(eps=3, min_samples=2).fit(X_embedded)
    print(clustering.labels_)
    
    #print("NMI:", normalized_mutual_info_score(clustering.labels_,encoded_labels))
    #print("ARI:", adjusted_rand_score(clustering.labels_,encoded_labels))
    
    #res = mclustpy(matrix, G=9, modelNames='EEE', random_seed=2020)
    




if __name__ == "__main__":
    # python generate_data.py
    # generate_data()
    # classify(csv_path='cond_mat.csv')
    args = parser.parse_args(sys.argv[1:])

    csv_path = args.csv_path
    classifiction_algorithm = args.classifiction_algorithm
    do_feature_selection = args.do_feature_selection
    feature_selection_algorithm = args.feature_selection_algorithm
    k_best = int(args.k_best)
    do_dimonsionality_reduction = args.do_dimonsionality_reduction
    n_components = int(args.n_components)
    test_ratio = float(args.test_ratio)
    key_column = args.key_column
    value_column = args.value_column
    min_df = int(args.min_df)
    shuffle = args.shuffle


    df = pd.read_csv("cond_mat.csv")[0:1000]
    clustering(df=df, filename="x.png",category_key='categories',text='title')
    #df = pd.read_csv(csv_path)
    #scikit_classify(df=df, key="categories", value="title_abstract")
