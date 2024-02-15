"""Module for clustering tasks."""
import pandas as pd

import umap
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import argparse

import sys
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from collections import Counter
from sklearn.mixture import GaussianMixture
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from time import time

import numpy as np

# from sklearn.ensemble import GradientBoostingClassifier




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
    "--key_column",
    default="categories",
    help="Column name for classes in csv file",
)

parser.add_argument(
    "--value_column",
    default="title_abstract",
    help="Column name for text in csv file",
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


def clustering(df=None, category_key="categories", text="title", filename=None,clustering_algorithm="KMeans", ndim=128):
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
    print(encoded_labels)
    matrix = np.nan_to_num(matrix, nan=0.0)
    print('matrix',matrix )
    X = reduce_dim(matrix, ndim)
    print('X',X )


    term_list = list(np.array(info).T[1])
    term_set = list(set(term_list))
    term_list = [term_set.index(term) for term in term_list]
    print("-------------------",clustering_algorithm)


    if(clustering_algorithm == "KMeans"):

      kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(X)
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

      print(f"Silhouette Coefficient: {metrics.silhouette_score(X, label):.3f}")
      X_embedded = TSNE2D(X)
      print('X_embedded',X_embedded )
      kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(X_embedded)
      label= kmeans.labels_
      print("labels")
      print(kmeans.labels_)
      centroids = kmeans.cluster_centers_
      u_labels = np.unique(label)


    
      #plotting the results:
      
      for i,k in zip(u_labels,term_list):
          plt.scatter(X_embedded[label == i , 0] , X_embedded[label == i , 1] , label=i)
      plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
      plt.legend()
      plt.savefig("clustering_result_"+clustering_algorithm+".pdf")
      plt.close()
      plt.show()


    elif (clustering_algorithm== "Mixture"):
        clustering = GaussianMixture(n_components=8, random_state=0).fit(X)
        label=clustering.predict(X)
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

        print(f"Silhouette Coefficient: {metrics.silhouette_score(X, label):.3f}")
    
        X_embedded = TSNE2D(X)
        print('X_embedded',X_embedded )
        #plotting the results:
        clustering = GaussianMixture(n_components=8, random_state=0).fit(X_embedded)
        label=clustering.predict(X_embedded)

        centroids = clustering.means_
        u_labels = np.unique(label)

        for i,k in zip(u_labels,term_list):
            plt.scatter(X_embedded[label == i , 0] , X_embedded[label == i , 1] , label=i)
        plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
        plt.legend()
        plt.savefig("clustering_result_"+clustering_algorithm+".pdf")
        plt.close()
        plt.show()


    elif (clustering_algorithm== "HDBSCAN"):
        reducer = umap.UMAP(n_components=20)
        reducer.fit(X)
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

        print(f"Silhouette Coefficient: {metrics.silhouette_score(X, label):.3f}")
    
        reducer = umap.UMAP(n_components=2)
        reducer.fit(X)
        X_embedded=reducer.embedding_
        hdb.fit(X_embedded)
        label=hdb.labels_
        u_labels = np.unique(label)

        for i,k in zip(u_labels,term_list):
            plt.scatter(X_embedded[label == i , 0] , X_embedded[label == i , 1] , label=i)
        plt.legend()
        plt.savefig("clustering_result_"+clustering_algorithm+".pdf")
        plt.close()
        plt.show()

    elif (clustering_algorithm== "DBSCAN"):
        reducer = umap.UMAP(n_components=30)
        reducer.fit(X)
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

        print(f"Silhouette Coefficient: {metrics.silhouette_score(X, label):.3f}")
    

        X_embedded = TSNE2D(X)
        X_embedded=reducer.embedding_
        clustering = DBSCAN( min_samples=10).fit(X_embedded)

        label=clustering.labels_
        u_labels = np.unique(label)

        for i,k in zip(u_labels,term_list):
            plt.scatter(X_embedded[label == i , 0] , X_embedded[label == i , 1] , label=i)
        plt.legend()
        plt.savefig("clustering_result_"+clustering_algorithm+".pdf")
        plt.close()
        plt.show()









if __name__ == "__main__":
    # python generate_data.py
    # generate_data()
    # classify(csv_path='cond_mat.csv')
    args = parser.parse_args(sys.argv[1:])

    csv_path = args.csv_path
    clustering_algorithm = args.clustering_algorithm

    key_column = args.key_column
    value_column = args.value_column



    df = pd.read_csv("pubchem.csv")[0:1000]
    clustering(df=df, filename="x.png",category_key='label_name',text='title',clustering_algorithm=clustering_algorithm)
    #df = pd.read_csv(csv_path)
    #scikit_classify(df=df, key="categories", value="title_abstract")
