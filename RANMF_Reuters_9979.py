import numpy as np
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import sparse
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.cluster import SpectralClustering
from sklearn.cluster import SpectralBiclustering
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances


def fetching_data_set(categories):
    files = reuters.fileids(categories=categories)
    data = []
    target = []
    # Selecting documents with multi labels
    for i in files:
        for f in categories:
            if i in reuters.fileids(categories=f):
                data.append(reuters.raw(fileids=i))
                target.append(f)

    # Selecting documents with single label
    # for i in files:
    #     flag = 0
    #     for f in categories:
    #         if i in reuters.fileids(categories=f):
    #             flag = flag + 1
    #             cat = f
    #     if flag == 1:
    #         data.append(reuters.raw(fileids=i))
    #         target.append(cat)

    print('Number of documents: ', len(data), len(target))
    # print(target.count(categories[0]), target.count(categories[1]))
    # print(target.count(categories[2]), target.count(categories[3]))
    # print(target.count(categories[4]), target.count(categories[5]))
    # print(target.count(categories[6]), target.count(categories[7]))
    # print(target.count(categories[8]), target.count(categories[9]))
    return data, target


def pre_processing(data):
    porter = PorterStemmer()
    vectorizer = TfidfVectorizer(stop_words='english')

    # Aggregating and vectorizing all texts
    all_text = [" ".join(data)]
    X = vectorizer.fit_transform(all_text)
    print('Dimension of vectorizing aggregated texts without stemming: ', X.shape)

    # # Opening dictionary
    # f = open('words.txt', 'r')
    # dictionary = f.read().split()
    # f.close()

    # # Filtering terms via dictionary and stemming
    # token_words = terms
    # print('All terms: ', len(token_words))
    # # stem_sentence = [porter.stem(word) for word in token_words if word in dictionary]
    # stem_sentence = [porter.stem(word) for word in token_words]
    #
    # all_text_stem = [" ".join(stem_sentence)]
    # print('Number of terms after filtering dictionary: ', len(stem_sentence))
    #
    # # Vectorizing aggregated texts after filtering and stemming
    # X = vectorizer.fit_transform(all_text_stem)
    # print('Dimension of vectorizing after filtering and stemming: ', X.shape)

    # Stemming texts separately and vectorizing
    texts_stem = []
    target = []
    for i in range(len(data)):
        token_words = word_tokenize(data[i])
        stem_sentence = []
        for word in token_words:
            stem_sentence.append(porter.stem(word.lower()))
        texts_stem.append(" ".join(stem_sentence))


    X = vectorizer.fit_transform(texts_stem)
    print('Dimension of tf-idf matrix ', X.shape)

    return X


def feature_selection(X, features_number):
    # Feature selection using mutual information
    X_new = SelectKBest(mutual_info_classif, k=features_number).fit_transform(X, target)
    print('Dimension of tf-idf matrix after mutual information: ', X_new.shape)
    return X_new


def removing_zero_rows(X_new, target):
    # Removing zero rows
    target_new = target
    X_new2 = X_new.todense()
    l = np.where(X_new2.sum(axis=1) != 0)[0]
    X_new = X_new2[l, :]
    index = []
    c = 0
    j = 0
    for i in range(X_new2.shape[0]):
        if X_new2[i].sum() == 0:
            target_new = np.delete(target_new, j)
            index.append(j)
            j = j - 1
            c = c + 1
        j = j + 1
    X_new = sparse.csr_matrix(X_new)
    print('The number of removed documents: ', c, X_new.shape)
    return X_new, target_new


def compared_methods(clusters, X_new, target_new):
    # Clustering
    kmeans = KMeans(n_clusters=clusters, max_iter=200).fit(X_new)
    print('KMeans NMI: ', normalized_mutual_info_score(kmeans.labels_, target_new))
    print('KMeans AMI: ', adjusted_mutual_info_score(kmeans.labels_, target_new))
    print('KMeans Silhouette: ', silhouette_score(X_new, kmeans.labels_))

    model = NMF(n_components=clusters, init='random')
    D = model.fit_transform(X_new)
    W = model.components_
    NMF_labels = np.argmax(D, axis=1)
    print('NMF NMI: ', normalized_mutual_info_score(NMF_labels, target_new))
    print('NMF AMI: ', adjusted_mutual_info_score(NMF_labels, target_new))
    print('NMF Silhouette: ', silhouette_score(X_new, NMF_labels))

    model = NMF(n_components=clusters, init='random', solver='mu', beta_loss='kullback-leibler')
    D = model.fit_transform(X_new)
    W = model.components_
    NMF_KL_labels = np.argmax(D, axis=1)
    print('NMF_KL NMI: ', normalized_mutual_info_score(NMF_KL_labels, target_new))
    print('NMF_KL AMI: ', adjusted_mutual_info_score(NMF_KL_labels, target_new))
    print('NMF_KL Silhouette: ', silhouette_score(X_new, NMF_KL_labels))

    Spectral = SpectralClustering(n_clusters=clusters, random_state=0).fit(X_new)
    print('Spectral NMI: ', normalized_mutual_info_score(Spectral.labels_, target_new))
    print('Spectral AMI: ', adjusted_mutual_info_score(Spectral.labels_, target_new))
    print('Spectral Silhouette: ', silhouette_score(X_new, Spectral.labels_))

    SpectralBi = SpectralBiclustering(n_clusters=clusters).fit(X_new)
    print('SpectralBi NMI: ', normalized_mutual_info_score(SpectralBi.row_labels_, target_new))
    print('SpectralBi AMI: ', adjusted_mutual_info_score(SpectralBi.row_labels_, target_new))
    print('SpectralBi Silhouette: ', silhouette_score(X_new, SpectralBi.row_labels_))

    SpectralCo = SpectralCoclustering(n_clusters=clusters).fit(X_new)
    print('SpectralCo NMI: ', normalized_mutual_info_score(SpectralCo.row_labels_, target_new))
    print('SpectralCo AMI: ', adjusted_mutual_info_score(SpectralCo.row_labels_, target_new))
    print('SpectralCo Silhouette: ', silhouette_score(X_new, SpectralCo.row_labels_))

    Agglomerative = AgglomerativeClustering(n_clusters=clusters).fit(X_new.todense())
    print('Agglomerative NMI: ', normalized_mutual_info_score(Agglomerative.labels_, target_new))
    print('Agglomerative AMI: ', adjusted_mutual_info_score(Agglomerative.labels_, target_new))
    print('Agglomerative Silhouette: ', silhouette_score(X_new, Agglomerative.labels_))

    brc = Birch(n_clusters=clusters).fit(X_new)
    labels = brc.predict(X_new)
    print('Birch NMI: ', normalized_mutual_info_score(labels, target_new))
    print('Birch AMI: ', adjusted_mutual_info_score(labels, target_new))
    print('Birch Silhouette: ', silhouette_score(X_new, labels))

    kmeansp = MiniBatchKMeans(n_clusters=clusters, max_iter=200).fit(X_new.todense())
    print('MiniBatchKMeans NMI: ', normalized_mutual_info_score(kmeansp.labels_, target_new))
    print('MiniBatchKMeans AMI: ', adjusted_mutual_info_score(kmeansp.labels_, target_new))
    print('MiniBatchKMeans Silhouette: ', silhouette_score(X_new, kmeansp.labels_))


def similarity(X_new):
    # Computing similarity
    S = sparse.csr_matrix(cosine_similarity(X_new))
    return S


def RANMF(clusters, X_new, target_new, S):
    #### The Proposed RANMF
    Lambeda = 0.05
    sumS = np.squeeze(np.array(S.sum(axis=1)))
    D = np.random.rand(X_new.shape[0], clusters)
    W = np.random.rand(clusters, X_new.shape[1])
    for ite in range(200):
        # print('Iteration: ', ite)
        DW = D @ W
        SD = S @ D

        DW[DW <= .000001] = .000001
        XDW = X_new / DW
        XDWW = XDW @ W.transpose()
        sumW = W.sum(axis=1)
        M = []
        for f in range(clusters):
            M.append(sumW[f] + (2 * Lambeda * D[:, f] * sumS))
        M = np.squeeze(M).transpose()
        M[M == 0] = .000001
        D = D * np.asarray(XDWW + (2 * Lambeda * SD)) / M
        ####################
        DW = D @ W
        DW[DW <= .000001] = .000001
        XDW = X_new / DW
        XDWD = D.transpose() @ XDW
        sumD = D.sum(axis=0)
        W = W * np.asarray(XDWD)
        T = []
        for f in range(clusters):
            if sumD[f] != 0:
                T.append(W[f, :] / sumD[f])
        T = np.squeeze(T)
        W = T

    RANMF_labels = np.argmax(D, axis=1)
    print('RANMF NMI: ', normalized_mutual_info_score(RANMF_labels, target_new))
    print('RANMF AMI: ', adjusted_mutual_info_score(RANMF_labels, target_new))
    print('RANMF Silhouette: ', silhouette_score(X_new, RANMF_labels))


def RANMF_converge(clusters, X_new, S):
    #### The Proposed RANMF
    Lambeda = 0.05
    converge = []
    before = 0
    X_new2 = X_new
    X_new2[X_new2 == 0] = .000001
    X_new2 = np.array(X_new2)
    sumS = np.squeeze(np.array(S.sum(axis=1)))
    D = np.random.rand(X_new.shape[0], clusters)
    W = np.random.rand(clusters, X_new.shape[1])
    for ite in range(200):
        # print('Iteration: ', ite)
        DW = D @ W
        SD = S @ D

        DW[DW <= .000001] = .000001
        XDW = X_new / DW
        DW = 0
        XDWW = XDW @ W.transpose()
        sumW = W.sum(axis=1)
        M = []
        for f in range(clusters):
            M.append(sumW[f] + (2 * Lambeda * D[:, f] * sumS))
        M = np.squeeze(M).transpose()
        M[M == 0] = .000001
        D = D * np.asarray(XDWW + (2 * Lambeda * SD)) / M
        ####################
        DW = D @ W
        DW[DW <= .000001] = .000001
        XDW = X_new / DW
        XDWD = D.transpose() @ XDW
        sumD = D.sum(axis=0)
        W = W * np.asarray(XDWD)
        T = []
        for f in range(clusters):
            if sumD[f] != 0:
                T.append(W[f, :] / sumD[f])
        T = np.squeeze(T)
        W = T

        # # Convergence
        DW[DW < .000001] = .000001
        cost = (X_new2 * np.array(np.log(X_new2 / DW)) - X_new2 + DW).sum()
        # print('Iteration: ', ite)
        # cost = cost + Lambeda * ((euclidean_distances(D, D) ** 2) * np.array(S.todense())).sum()

        # if (before - cost) < 0:
        # print(' Not converge: ', ite)
        before = cost
        print(cost)
        converge.append(cost)


clusters = 10
features_number = 2000

categories = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']

data, target = fetching_data_set(categories[:4])
X = pre_processing(data)

# # With feature selection
X_new = feature_selection(X, features_number)

# # Without feature selection
# X_new = X

X_new, target_new = removing_zero_rows(X_new, target)

compared_methods(clusters, X_new, target_new)

S = similarity(X_new)

RANMF(clusters, X_new, target_new, S)

RANMF_converge(clusters, X_new, S)
