# adapted from https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Arturo Amor <david-arturo.amor-quiroz@inria.fr>
# License: BSD 3 clause
import numpy as np
from time import time
from collections import defaultdict
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def fit_and_evaluate(km, X, name=None, n_runs=5):
    name = km.__class__.__name__ if name is None else name

    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    train_times = np.asarray(train_times)

    print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)

if __name__ == "__main__" :
    evaluations = []
    evaluations_std = []

    categories = [
        'alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc'
    ]

    dataset = fetch_20newsgroups(
        remove=("headers", "footers", "quotes"),
        subset="all",
        categories=categories,
        shuffle=True,
        random_state=42,
    )

    labels = dataset.target
    unique_labels, category_sizes = np.unique(labels, return_counts=True)
    true_k = unique_labels.shape[0]

    print(f"{len(dataset.data)} documents - {true_k} categories")


    vectorizer = TfidfVectorizer(
        max_df=0.5,min_df=5, stop_words="english",)
    t0 = time()
    X_tfidf = vectorizer.fit_transform(dataset.data)

    print(f"vectorization done in {time() - t0:.3f} s")
    print(f"n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}")



    tfidf_kmeans = KMeans(n_clusters=true_k,max_iter=100, n_init=5,)

    fit_and_evaluate(tfidf_kmeans, X_tfidf, name="KMeans\non tf-idf vectors")
    print()

    lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    t0 = time()
    X_lsa = lsa.fit_transform(X_tfidf)
    explained_variance = lsa[0].explained_variance_ratio_.sum()

    print(f"LSA done in {time() - t0:.3f} s")
    print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")

    lsa_kmeans = KMeans(n_clusters=true_k, max_iter=100, n_init=1,)

    fit_and_evaluate(lsa_kmeans, X_lsa, name="KMeans\nwith LSA on tf-idf vectors")