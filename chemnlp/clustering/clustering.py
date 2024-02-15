"""Module for clustering tasks."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

import umap

import matplotlib.pyplot as plt

import argparse

import sys
from sklearn.decomposition import TruncatedSVD



import pandas as pd


import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import argparse


import sys
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN

from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from time import time

import numpy as np





parser = argparse.ArgumentParser(description="ChemNLP package.")
parser.add_argument(
    "--csv_path",
    default="filename.csv",
    help="Path for comma separated file.",
)

parser.add_argument(
    "--clustering_algorithm",
    default="KMeans",

)

parser.add_argument(
    "--n_clusters",
    default=8,
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
    default="abstract",
    help="Column name for text in csv file",
)

parser.add_argument(
    "--min_df",
    default=5,
    help="Minimum numbers of documents a word must be present in to be kept",
)



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

def clustering(
    csv_path=None,
    key="categories",
    value="abstract",
    test_size=0.2,
    min_df=5,
    ngram_range=(1,2),
    print_common=False,
    model=None,
    categorize=True,  # False is buggy, need to check
    shuffle=False,
    clustering_algorithm = "SVM",
    n_clusters=8,
):
    """Classifcy data using scikit-learn library algos."""
    df = pd.read_csv(csv_path, dtype="str")
    df = df[0:1000]
    # df=pd.read_csv(csv_path,dtype={'id':'str'})
    if categorize:
        df["category_id"] = df[key].factorize()[0]
        category_id_df = (
            df[[key, "category_id"]].sort_values("category_id")
            # df[[key, "category_id"]].drop_duplicates()
            # .sort_values("category_id")
        )

        category_to_id = dict(category_id_df.values)
        print("category_to_id", category_to_id)
    print(df)
    print(df[key].value_counts())
    abstract_tfidf = TfidfVectorizer(
        sublinear_tf=True,
        # logarithmic form for frequency
        min_df=min_df,
        # minimum numbers of documents a word must be present in to be kept
        norm="l2",
        # norm is set to l2, to ensure all our feature
        # vectors have a euclidian norm of 1
        encoding="latin-1",  # utf-8 possible
        ngram_range=ngram_range,
        # (1, 2) to indicate that we want to consider both unigrams and bigrams
        stop_words="english",
        # remove common pronouns ("a", "the", ...)
        # to reduce the number of noisy features
    )
    # Abstact features:
    abstract_features = abstract_tfidf.fit_transform(
        df[value]
    )  # .toarray()  # abstract
    if categorize:
        encoded_labels = df.category_id  # categories (cond-mat)
    else:
        encoded_labels = df[key]  # categories (cond-mat)
    # encoded_labels = df.category_id  # categories (cond-mat)
    print(
        "abstract_features.shape", abstract_features.shape
    )  # abstracts represented by features,
    # representing tf-idf score for different unigrams and bigrams
   
 
 
    print("-------------------",clustering_algorithm)


    if(clustering_algorithm == "KMeans"):

      kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(abstract_features)
      label= kmeans.labels_
      print("labels")
      print(kmeans.labels_)
      print("Normalized Mutual Information:", normalized_mutual_info_score(encoded_labels,kmeans.labels_))
      print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(encoded_labels, label):.3f}")
      print(
          "Adjusted Mutual Information:"
          f" {metrics.adjusted_mutual_info_score(encoded_labels, label):.3f}"
      )
      print(f"Homogeneity: {metrics.homogeneity_score(encoded_labels, label):.3f}")
      print(f"Completeness: {metrics.completeness_score(encoded_labels, label):.3f}")
      print(f"V-measure: {metrics.v_measure_score(encoded_labels, label):.3f}")

      print(f"Silhouette Coefficient: {metrics.silhouette_score(abstract_features, label):.3f}")
      X_embedded = reduce_dim(abstract_features,2)
      print('X_embedded',X_embedded )
      kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(X_embedded)
      label= kmeans.labels_
      print("labels")
      print(kmeans.labels_)
      centroids = kmeans.cluster_centers_
      u_labels = np.unique(label)


    
      #plotting the results:
      
      for i in u_labels:
          plt.scatter(X_embedded[label == i , 0] , X_embedded[label == i , 1] , label=i)
      plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
      plt.legend()
      plt.savefig("clustering_result_"+clustering_algorithm+".pdf")
      plt.close()
      plt.show()


    elif (clustering_algorithm== "Mixture"):
        abstract_features = reduce_dim(abstract_features,128)
        clustering = GaussianMixture(n_components=8, random_state=0).fit(abstract_features)
        label=clustering.predict(abstract_features)
        print("labels")
        print(label)
        centroids = clustering.means_
        u_labels = np.unique(label)
        
        
        print("Normalized Mutual Information:", normalized_mutual_info_score(encoded_labels,label))
        print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(encoded_labels, label):.3f}")
        print(
            "Adjusted Mutual Information:"
            f" {metrics.adjusted_mutual_info_score(encoded_labels, label):.3f}"
        )
        print(f"Homogeneity: {metrics.homogeneity_score(encoded_labels, label):.3f}")
        print(f"Completeness: {metrics.completeness_score(encoded_labels, label):.3f}")
        print(f"V-measure: {metrics.v_measure_score(encoded_labels, label):.3f}")

        print(f"Silhouette Coefficient: {metrics.silhouette_score(abstract_features, label):.3f}")
    
        X_embedded = reduce_dim(abstract_features,2)
        print('X_embedded',X_embedded )
        #plotting the results:
        clustering = GaussianMixture(n_components=8, random_state=0).fit(X_embedded)
        label=clustering.predict(X_embedded)

        centroids = clustering.means_
        u_labels = np.unique(label)

        for i in u_labels:
            plt.scatter(X_embedded[label == i , 0] , X_embedded[label == i , 1] , label=i)
        plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
        plt.legend()
        plt.savefig("clustering_result_"+clustering_algorithm+".pdf")
        plt.close()
        plt.show()


    elif (clustering_algorithm== "HDBSCAN"):
        abstract_features = reduce_dim(abstract_features, 128)
        reducer = umap.UMAP(n_components=20)
        reducer.fit(abstract_features)
        X_embedded=reducer.embedding_
        hdb = HDBSCAN(min_cluster_size=20, min_samples=5)
        hdb.fit(X_embedded)
        label=hdb.labels_

        #clustering = DBSCAN(eps=3, min_samples=2).fit(X)
        #print("labels")
        #print(clustering.labels_)
        #label=clustering.labels_
        
        print("labels")
        print(label)


        print("Normalized Mutual Information:", normalized_mutual_info_score(encoded_labels,label))
        print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(encoded_labels, label):.3f}")
        print(
            "Adjusted Mutual Information:"
            f" {metrics.adjusted_mutual_info_score(encoded_labels, label):.3f}"
        )
        print(f"Homogeneity: {metrics.homogeneity_score(encoded_labels, label):.3f}")
        print(f"Completeness: {metrics.completeness_score(encoded_labels, label):.3f}")
        print(f"V-measure: {metrics.v_measure_score(encoded_labels, label):.3f}")

        print(f"Silhouette Coefficient: {metrics.silhouette_score(abstract_features, label):.3f}")
    
        reducer = umap.UMAP(n_components=2)
        reducer.fit(abstract_features)
        X_embedded=reducer.embedding_
        hdb.fit(X_embedded)
        label=hdb.labels_
        u_labels = np.unique(label)

        for i in u_labels:
            plt.scatter(X_embedded[label == i , 0] , X_embedded[label == i , 1] , label=i)
        plt.legend()
        plt.savefig("clustering_result_"+clustering_algorithm+".pdf")
        plt.close()
        plt.show()

    elif (clustering_algorithm== "DBSCAN"):
        abstract_features = reduce_dim(abstract_features, 128)
        reducer = umap.UMAP(n_components=30)
        reducer.fit(abstract_features)
        X_embedded=reducer.embedding_


        clustering = DBSCAN(min_samples=10).fit(X_embedded)
        print("labels")
        print(clustering.labels_)
        label=clustering.labels_
        

        u_labels = np.unique(label)

        print("Normalized Mutual Information:", normalized_mutual_info_score(encoded_labels,label))
        print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(encoded_labels, label):.3f}")
        print(
            "Adjusted Mutual Information:"
            f" {metrics.adjusted_mutual_info_score(encoded_labels, label):.3f}"
        )
        print(f"Homogeneity: {metrics.homogeneity_score(encoded_labels, label):.3f}")
        print(f"Completeness: {metrics.completeness_score(encoded_labels, label):.3f}")
        print(f"V-measure: {metrics.v_measure_score(encoded_labels, label):.3f}")

        print(f"Silhouette Coefficient: {metrics.silhouette_score(abstract_features, label):.3f}")
    

        X_embedded = reduce_dim(abstract_features,2)
        X_embedded=reducer.embedding_
        clustering = DBSCAN( min_samples=10).fit(X_embedded)

        label=clustering.labels_
        u_labels = np.unique(label)

        for i in u_labels:
            plt.scatter(X_embedded[label == i , 0] , X_embedded[label == i , 1] , label=i)
        plt.legend()
        plt.savefig("clustering_result_"+clustering_algorithm+".pdf")
        plt.close()
        plt.show()




if __name__ == "__main__":
    # python generate_data.py
    # generate_data()
    # clustering(csv_path='cond_mat.csv')
    args = parser.parse_args(sys.argv[1:])

    csv_path = args.csv_path
    clustering_algorithm = args.clustering_algorithm

    n_clusters = int(args.n_clusters)
    test_ratio = float(args.test_ratio)
    key_column = args.key_column
    value_column = args.value_column
    min_df = int(args.min_df)


    clustering(
        csv_path=csv_path,
        test_size=test_ratio,
        key=key_column,
        value=value_column,
        min_df=min_df,
        clustering_algorithm = clustering_algorithm,

     
        n_clusters=n_clusters,
    )
    #df = pd.read_csv(csv_path)
    #scikit_clustering(df=df, key="categories", value="abstract_abstract")


