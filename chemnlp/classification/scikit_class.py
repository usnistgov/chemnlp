"""Module for classification tasks."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import numpy as np
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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import xgboost as xgb
import sys
from sklearn.decomposition import TruncatedSVD

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



def sk_class(
    csv_path=None,
    key="categories",
    value="title_abstract",
    test_size=0.2,
    min_df=5,
    ngram_range=[1, 2],
    print_common=False,
    model=None,
    categorize=True,  # False is buggy, need to check
    shuffle=False,
):
    """Classifcy data using scikit-learn library algos."""
    df = pd.read_csv(csv_path, dtype="str")
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
    title_tfidf = TfidfVectorizer(
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
    # Title features:
    title_features = title_tfidf.fit_transform(
        df[value]
    )  # .toarray()  # title
    if categorize:
        title_labels = df.category_id  # categories (cond-mat)
    else:
        title_labels = df[key]  # categories (cond-mat)
    # title_labels = df.category_id  # categories (cond-mat)
    print(
        "title_features.shape", title_features.shape
    )  # titles represented by features,
    # representing tf-idf score for different unigrams and bigrams
    if model is None:
        model = LogisticRegression(random_state=0)  # LinearSVC()
    (
        X_train,
        X_test,
        y_train,
        y_test,
        indices_train,
        indices_test,
    ) = train_test_split(
        title_features,
        title_labels,
        df.index,
        test_size=test_size,
        random_state=0,
        shuffle=shuffle,
    )
    if not categorize:
        category_to_id = list(set(df[key].values))
    id_to_category = dict((v, k) for k, v in category_to_id.items())
    info = {}
    train_info = {}
    test_info = {}
    for i, j in zip(df.id[indices_train], y_train):
        train_info[i] = id_to_category[int(j)]  # j
    for i, j in zip(df.id[indices_test], y_test):
        test_info[i] = id_to_category[int(j)]  # j
    info["train"] = train_info
    info["test"] = test_info
    dumpjson(data=info, filename="arXiv_categories.json")
    #model.fit(X_train, y_train)
    #y_pred = model.predict(X_test)
    f = open("pred_test.csv", "w")
    # f.write(str(accuracy_score(y_test, y_pred)))
    # f.close()
    line = "id,target,prediction\n"
    f.write(line)
    # pickle.dump(model, open("log.pk", "wb"))

    # print ('indices_train',df.id[indices_train]
    # .astype(str),len(indices_train))
    # print('indices_test',df.id[indices_test].astype(str),len(indices_test))
    #print("Logistic model accuracy", accuracy_score(y_test, y_pred))

    #svd = TruncatedSVD(n_components=15, random_state=42)
    #title_features = svd.fit_transform(title_features)     




   # print("Training SVC:")
   # model = LinearSVC(verbose=True)
   # model.fit(X_train, y_train)
   # y_pred = model.predict(X_test)
    #print("SVC", accuracy_score(y_test, y_pred))


   # Perform chi-square feature selection
    k_best = 2000  # Number of features to select
    
    #chi2_selector = SelectKBest(chi2, k=k_best)
    #chi2_selector = SelectKBest(mutual_info_classif, k=k_best)
    #chi2_selector = SelectKBest(f_classif, k=k_best)
    #X_train = chi2_selector.fit_transform(X_train, y_train)
    #X_test = chi2_selector.transform(X_test)

 
    print("Training MLPClassifier:")
    # MLPClassifier()

    model = MLPClassifier(solver='adam',activation='relu', alpha=1e-2, hidden_layer_sizes=(17), random_state=1,learning_rate='adaptive')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f = open("accuracy_MLPClassifier", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()
    print("MLPClassifier", accuracy_score(y_test, y_pred))





    print("Training MLPClassifier2:")
    # MLPClassifier()

    model = MLPClassifier(hidden_layer_sizes=(150,100,50),
                        max_iter = 300,activation = 'relu',
                        solver = 'adam')

    model.fit(trainX_scaled, y_train)

    y_pred = model.predict(X_test)
    f = open("accuracy_MLPClassifier2", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()
    print("MLPClassifier2", accuracy_score(y_test, y_pred))


    print("Training KNN:")
   
    #chi2_selector = SelectKBest(chi2, k=k_best)
    #chi2_selector = SelectKBest(mutual_info_classif, k=k_best)
    chi2_selector = SelectKBest(f_classif, k=k_best)
    X_train_knn = chi2_selector.fit_transform(X_train, y_train)
    X_test_knn = chi2_selector.transform(X_test)

 

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_knn, y_train)
    y_pred = model.predict(X_test_knn)
    f = open("accuracy_KNN", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()
    print("KNN", accuracy_score(y_test, y_pred))







    print("Training RandomForestClassifier:")
    # RandomForestClassifier()

    model = RandomForestClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("RandomForestClassifier", accuracy_score(y_test, y_pred))


    print("Training XGBoost:")
    # XGBoost()

    xgb.set_config(verbosity=2)     
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    print("XGBoost", accuracy_score(y_test, y_pred))




    print("Training logistic regression:")
    # LinearSVC()
    model = LogisticRegression(random_state=0,verbose=1)  
  
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f = open("accuracy_logistic", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()
    print("Logistic", accuracy_score(y_test, y_pred))








    print("Training MultiNomial:")
    # GradientBoostingClassifier()
    model = MultinomialNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f = open("accuracy_MultinomialNB", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()


    print("MultinomialNB", accuracy_score(y_test, y_pred))




def scikit_classify(
    df=None,
    key="term",
    value="title",
    CV=3,
    test_size=0.2,
    min_df=5,
    ngram_range=[1, 2],
    print_common=False,
    model_selection=False,
):
    """Classifcy data using scikit-learn library algos."""
    print(df)
    print(df[key].value_counts())
    df["category_id"] = df[key].factorize()[0]
    category_id_df = (
        df[[key, "category_id"]].drop_duplicates().sort_values("category_id")
    )

    category_to_id = dict(category_id_df.values)
    # id_to_category = dict(category_id_df[["category_id", key]].values)

    print(df.head())

    title_tfidf = TfidfVectorizer(
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

    # Title features:
    title_features = title_tfidf.fit_transform(
        df[value]
    )  # .toarray()  # title
    title_labels = df.category_id  # categories (cond-mat)
    print(
        "title_features.shape", title_features.shape
    )  # titles represented by features,
    # representing tf-idf score for different unigrams and bigrams

    if print_common:
        N = 2  # number of correlated items to display
        for title, category_id in sorted(category_to_id.items()):
            features_chi2 = chi2(title_features, title_labels == category_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(title_tfidf.get_feature_names_out())[
                indices
            ]
            unigrams = [v for v in feature_names if len(v.split(" ")) == 1]
            bigrams = [v for v in feature_names if len(v.split(" ")) == 2]
            print("# '{}':".format(title))
            print(
                "  . Most correlated unigrams:\n. {}".format(
                    "\n. ".join(unigrams[-N:])
                )
            )
            print(
                "  . Most correlated bigrams:\n. {}".format(
                    "\n. ".join(bigrams[-N:])
                )
            )

    #svd = TruncatedSVD(n_components=15, random_state=42)
    #title_features = svd.fit_transform(title_features)     


    (
        X_train,
        X_test,
        y_train,
        y_test,
        indices_train,
        indices_test,
    ) = train_test_split(
        title_features,
        title_labels,
        df.index,
        test_size=test_size,
        random_state=0,
        shuffle=shuffle,
    )

    print("Training SVC:")
    model = LinearSVC(verbose=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("SVC", accuracy_score(y_test, y_pred))


   # Perform chi-square feature selection
    k_best = 1500  # Number of features to select
    
    #chi2_selector = SelectKBest(chi2, k=k_best)
    chi2_selector = SelectKBest(mutual_info_classif, k=k_best)
    X_train = chi2_selector.fit_transform(X_train, y_train)
    X_test = chi2_selector.transform(X_test)

 

    print("Training XGBoost:")
    # XGBoost()

    xgb.set_config(verbosity=2)     
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    print("XGBoost", accuracy_score(y_test, y_pred))



    print("Training MLPClassifier:")
    # MLPClassifier()

    model = MLPClassifier(solver='adam',activation='relu', alpha=1e-2, hidden_layer_sizes=(5, 2), random_state=1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f = open("accuracy_MLPClassifier", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()
    print("MLPClassifier", accuracy_score(y_test, y_pred))



    print("Training logistic regression:")
    # LinearSVC()
    model = LogisticRegression(random_state=0,verbose=1)  
  
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f = open("accuracy_logistic", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()
    print("Logistic", accuracy_score(y_test, y_pred))


    print("Training KNN:")
   

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f = open("accuracy_KNN", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()
    print("KNN", accuracy_score(y_test, y_pred))







    print("Training MultiNomial:")
    # GradientBoostingClassifier()
    model = MultinomialNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f = open("accuracy_MultinomialNB", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()


    print("MultinomialNB", accuracy_score(y_test, y_pred))




if __name__ == "__main__":
    # python generate_data.py
    # generate_data()
    # sk_class(csv_path='cond_mat.csv')
    args = parser.parse_args(sys.argv[1:])
    csv_path = args.csv_path
    test_ratio = float(args.test_ratio)
    key_column = args.key_column
    value_column = args.value_column
    min_df = int(args.min_df)
    shuffle = args.shuffle

    sk_class(
        csv_path=csv_path,
        test_size=test_ratio,
        key=key_column,
        value=value_column,
        min_df=min_df,
        shuffle=shuffle,
    )
    #df = pd.read_csv(csv_path)
    #scikit_classify(df=df, key="categories", value="title_abstract")
