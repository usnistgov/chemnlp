"""Module for classification tasks."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from jarvis.db.figshare import data
import pickle
from sklearn.metrics import accuracy_score


def generate_data(n_entries=None, filename="cond_mat.csv"):
    """Generate cond-mat dataset."""
    d = data("arXiv")
    df = pd.DataFrame(d)
    cond_mat_topics = [
        "cond-mat.mtrl-sci",
        "cond-mat.mes-hall",
        "cond-mat.str-el",
        "cond-mat.stat-mech",
        "cond-mat.supr-con",
        "cond-mat.soft",
        "cond-mat.quant-gas",
        "cond-mat.other",
        "cond-mat.dis-nn",
    ]
    df_cond_mat = df[
        df["categories"].apply(
            lambda x: "cond-mat" in x and x in cond_mat_topics
        )
    ]
    if n_entries is not None:
        df_cond_mat = df_cond_mat[0:n_entries]
    dff = df_cond_mat[["id", "title", "abstract", "categories"]]
    dff["title_abstract"] = dff["title"] + " " + df["abstract"]
    dff.to_csv(filename)


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
    if model_selection:
        models = [
            # GradientBoostingClassifier(),
            RandomForestClassifier(),
            # n_estimators=200, max_depth=3, random_state=0),
            LinearSVC(),
            MultinomialNB(),
            LogisticRegression(random_state=0),
            # try: multi_class='multinomial',
            # solver='lbfgs'...same accuracy actually
        ]

        cv_df = pd.DataFrame(index=range(CV * len(models)))
        entries = []
        for model in models:
            model_name = model.__class__.__name__
            accuracies = cross_val_score(
                model, title_features, title_labels, scoring="accuracy", cv=CV
            )  # features: ; labels:
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(
            entries, columns=["model_name", "fold_idx", "accuracy"]
        )

        print(cv_df)
        # Visualize model accuracy with box plot and strip plot:

        plt.figure(figsize=(10, 8))
        plt.rcParams.update({"font.size": 18})
        sns.boxplot(x="model_name", y="accuracy", data=cv_df)
        sns.stripplot(
            x="model_name",
            y="accuracy",
            data=cv_df,
            size=8,
            jitter=True,
            edgecolor="gray",
            linewidth=2,
        )

        plt.ylim(0, 1)

        # plt.title('Model Benchmark Comparison from Title Text Data')
        plt.ylabel("Accuracy (cross-validation score)")
        # plt.xlabel('Model')
        plt.xlabel("")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("cv.pdf")
        plt.close()
    print("Training logistic regression:")
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
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f = open("accuracy_logistic", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()
    print("Logistic", accuracy_score(y_test, y_pred))
    pickle.dump(model, open("log.pk", "wb"))

    plt.rcParams.update({"font.size": 20})
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(
        conf_mat / conf_mat.sum(axis=1)[:, np.newaxis],
        annot=True,
        # fmt="d",
        fmt=".1%",
        cbar=False,
        square=True,
        cmap=sns.diverging_palette(20, 220, n=200),
        xticklabels=category_id_df[key].values,
        yticklabels=category_id_df[key].values,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("conf_log.pdf")
    plt.close()

    print("Training SVC:")
    model = LinearSVC()
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
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("SVC", accuracy_score(y_test, y_pred))
    pickle.dump(model, open("svc.pk", "wb"))

    plt.rcParams.update({"font.size": 20})
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(
        conf_mat / conf_mat.sum(axis=1)[:, np.newaxis],
        annot=True,
        # fmt="d",
        fmt=".1%",
        cbar=False,
        square=True,
        cmap=sns.diverging_palette(20, 220, n=200),
        xticklabels=category_id_df[key].values,
        yticklabels=category_id_df[key].values,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("conf_svc.pdf")
    plt.close()
    f = open("accuracy_svc", "w")
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()


if __name__ == "__main__":
    # python generate_data.py
    df = pd.read_csv("cond_mat.csv")
    scikit_classify(df=df, key="categories", value="title_abstract")
