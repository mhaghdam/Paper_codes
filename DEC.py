import numpy as np
from nltk.corpus import reuters
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment


def accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
            assert y_pred.size == y_true.size
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def autoencoder_f(dims: object, act: object = 'relu', init: object = 'glorot_uniform') -> object:
    n_stacks = len(dims) - 1

    input_data = Input(shape=(dims[0],), name='input')
    x = input_data

    # internal layers of encoder
    for i in range(n_stacks - 1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)  # latent hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)
    x = encoded
    # internal layers of decoder
    for i in range(n_stacks - 1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)  # decoder output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)

    decoded = x

    autoencoder_model = Model(inputs=input_data, outputs=decoded, name='autoencoder')
    encoder_model = Model(inputs=input_data, outputs=encoded, name='encoder')

    return autoencoder_model, encoder_model


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))  # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def fetching_dataset_reuters(categories):
    files = reuters.fileids(categories=categories)
    data = []
    target = []
    # Selecting documents with multi labels
    for i in files:
        for f in categories:
            if i in reuters.fileids(categories=f):
                data.append(reuters.raw(fileids=i))
                target.append(f)

    print('Number of documents: ', len(data), len(target))
    # print(target.count(categories[0]), target.count(categories[1]))
    # print(target.count(categories[2]), target.count(categories[3]))
    # print(target.count(categories[4]), target.count(categories[5]))
    # print(target.count(categories[6]), target.count(categories[7]))
    # print(target.count(categories[8]), target.count(categories[9]))
    return data, target


def fetching_dataset_20ng():
    data = []
    f = open("20ng-all-stemmed.txt", "r")
    for x in f:
        data.append(x)
    f.close()
    print('Number of documents: ', len(data))
    return data


def fetching_dataset_webKB():
    data = []
    f = open("webkb-all-stemmed.txt", "r")
    for x in f:
        data.append(x)
    f.close()
    print('Number of documents: ', len(data))
    return data


def pre_processing_reuters(data):
    porter = PorterStemmer()
    vectorizer = TfidfVectorizer(stop_words='english')

    # Aggregating and vectorizing all texts
    all_text = [" ".join(data)]
    X = vectorizer.fit_transform(all_text)
    print('Dimension of vectorizing aggregated texts without stemming: ', X.shape)

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


def pre_processing_20ng(data, categories):
    porter = PorterStemmer()
    vectorizer = TfidfVectorizer(stop_words='english')

    # Aggregating and vectorizing all texts
    all_text = [" ".join(data)]
    X = vectorizer.fit_transform(all_text)
    print('Dimension of vectorizing aggregated texts without stemming: ', X.shape)

    texts_stem = []
    target = []
    for i in range(len(data)):
        token_words = word_tokenize(data[i])
        stem_sentence = []
        if token_words[0] in categories:
            target.append("".join(token_words[0]))
            for word in token_words[1:]:
                stem_sentence.append(porter.stem(word.lower()))
            texts_stem.append(" ".join(stem_sentence))

    X = vectorizer.fit_transform(texts_stem)
    print('Dimension of tf-idf matrix ', X.shape)

    return X, target


def pre_processing_webKB(data):
    porter = PorterStemmer()
    vectorizer = TfidfVectorizer(stop_words='english')

    # Aggregating and vectorizing all texts
    all_text = [" ".join(data)]
    X = vectorizer.fit_transform(all_text)
    print('Dimension of vectorizing aggregated texts without stemming: ', X.shape)

    # Stemming texts separately and vectorizing
    texts_stem = []
    target = []
    for i in range(len(data)):
        token_words = word_tokenize(data[i])
        stem_sentence = []
        target.append("".join(token_words[0]))
        for word in token_words[1:]:
            stem_sentence.append(porter.stem(word.lower()))
        texts_stem.append(" ".join(stem_sentence))

    X = vectorizer.fit_transform(texts_stem)
    print('Dimension of tf-idf matrix ', X.shape)

    return X, target


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


def deep_learning_clustering(X_new, y):
    # For Reuters
    for index, item in enumerate(y):
        if item == 'earn':
            y[index] = 1
        elif item == 'acq':
            y[index] = 2
        elif item == 'money-fx':
            y[index] = 3
        elif item == 'grain':
            y[index] = 4
        elif item == 'crude':
            y[index] = 5
        elif item == 'trade':
            y[index] = 6
        elif item == 'interest':
            y[index] = 7
        elif item == 'ship':
            y[index] = 8
        elif item == 'wheat':
            y[index] = 9
        elif item == 'corn':
            y[index] = 10

    # ## For 20ng
    # for index, item in enumerate(y):
    #     if item == 'alt.atheism':
    #         y[index] = 1
    #     elif item == 'comp.graphics':
    #         y[index] = 2
    #     elif item == 'comp.os.ms-windows.misc':
    #         y[index] = 3
    #     elif item == 'comp.sys.ibm.pc.hardware':
    #         y[index] = 4
    #     elif item == 'comp.sys.mac.hardware':
    #         y[index] = 5
    #     elif item == 'comp.windows.x':
    #         y[index] = 6
    #     elif item == 'misc.forsale':
    #         y[index] = 7
    #     elif item == 'rec.autos':
    #         y[index] = 8
    #     elif item == 'rec.motorcycles':
    #         y[index] = 9
    #     elif item == 'rec.sport.baseball':
    #         y[index] = 10
    #     elif item == 'rec.sport.hockey':
    #         y[index] = 11
    #     elif item == 'sci.crypt':
    #         y[index] = 12
    #     elif item == 'sci.electronics':
    #         y[index] = 13
    #     elif item == 'sci.med':
    #         y[index] = 14
    #     elif item == 'sci.space':
    #         y[index] = 15
    #     elif item == 'soc.religion.christian':
    #         y[index] = 16
    #     elif item == 'talk.politics.guns':
    #         y[index] = 17
    #     elif item == 'talk.politics.mideast':
    #         y[index] = 18
    #     elif item == 'talk.politics.misc':
    #         y[index] = 19
    #     elif item == 'talk.religion.misc':
    #         y[index] = 20

    # ## For webKB
    # for index, item in enumerate(y):
    #     if item == 'course':
    #         y[index] = 1
    #     elif item == 'faculty':
    #         y[index] = 2
    #     elif item == 'project':
    #         y[index] = 3
    #     elif item == 'student':
    #         y[index] = 4

    print('Creating and Training K-means Model')
    n_epochs = 200
    batch_size = 128
    # dims = [X_new.shape[-1], 600, 600, 2000, 10]
    dims = [X_new.shape[-1], 300, 300, 1000, 10]
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    pretrain_epochs = n_epochs
    batch_size = batch_size
    # save_dir = '/media/sf_LINUX_FILES/DeepClustering'

    print('Start building Deep Model.')
    autoencoder, encoder = autoencoder_f(dims, init=init)
    X_new_List = X_new.todense().tolist()
    autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
    autoencoder.fit(x=X_new_List, y=X_new_List, batch_size=batch_size, epochs=pretrain_epochs)
    # autoencoder.save_weights(save_dir + '/ae_weights.h5')
    # autoencoder.load_weights(save_dir + '/ae_weights.h5')

    print('Add Clustering Layer.')
    clustering_layer = ClusteringLayer(clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

    # initialize cluster centers using k-means
    kmeans = KMeans(n_clusters=clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(X_new))
    y_pred_last = np.copy(y_pred)

    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    loss = 0
    index = 0
    maxiter = 800
    update_interval = 140
    index_array = np.arange(X_new.shape[0])
    tol = 0.001  # tolerance threshold to stop training
    y = np.array(y)
    print('Start training...')
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = model.predict(X_new, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)
            if X_new is not None:
                acc = np.round(accuracy(y, y_pred), 5)
                nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
                ari = np.round(adjusted_rand_score(y, y_pred), 5)
                loss = np.round(loss, 5)
                print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

            # check stop criterion - model convergence
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index + 1) * batch_size, X_new.shape[0])]
        loss = model.train_on_batch(x=X_new.todense()[idx], y=p[idx])
        index = index + 1 if (index + 1) * batch_size <= X_new.shape[0] else 0

    # model.save_weights(save_dir + '/DEC_model_final.h5')
    # Load the clustering model trained weights
    # model.load_weights(save_dir + '/DEC_model_final.h5')

    print('Predicting...')
    q = model.predict(X_new, verbose=0)
    p = target_distribution(q)  # update the auxiliary target distribution p

    # evaluate the clustering performance
    y_pred = q.argmax(1)

    print('Deep learning NMI: ', normalized_mutual_info_score(y_pred, y))
    print('Deep learning AMI: ', adjusted_mutual_info_score(y_pred, y))
    print('Deep learning Silhouette: ', silhouette_score(X_new, y_pred))


####################################
# For running reuters
categories = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']

clusters = 4
features_number = 2000

data, target = fetching_dataset_reuters(categories[:4])
X = pre_processing_reuters(data)

# # With feature selection
# X_new = feature_selection(X, features_number)

# # Without feature selection
X_new = X

X_new, target_new = removing_zero_rows(X_new, target)
deep_learning_clustering(X_new, target_new)

# ###################################################
# # For running 20ng
# clusters = 20
# features_number = 2000
# categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
#               'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
#               'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
#               'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
#               'talk.religion.misc']
#
# data = fetching_dataset_20ng()
# X, target = pre_processing_20ng(data, categories[:20])
#
# # # With feature selection
# X_new = feature_selection(X, features_number)
#
# # # Without feature selection
# # X_new = X
#
# X_new, target_new = removing_zero_rows(X_new, target)
# deep_learning_clustering(X_new, target_new)

##################################################
# For running webKB
# clusters = 4
# features_number = 500
#
# data = fetching_dataset_webKB()
# X, target = pre_processing_webKB(data)
#
# # # With feature selection
# # X_new = feature_selection(X, features_number)
#
# # # Without feature selection
# X_new = X
#
# X_new, target_new = removing_zero_rows(X_new, target)
# deep_learning_clustering(X_new, target_new)