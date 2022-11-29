"""
.. _gensim.models.doc2vec.TaggedDocument: https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.TaggedDocument
.. _gensim.models.doc2vec.Doc2Vec: https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec
.. _pandas.DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
.. _pandas.Series: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
.. _numpy.array: https://numpy.org/doc/stable/reference/generated/numpy.array.html
"""

import _pickle as pickle
import logging
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import psutil
import ray
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.neighbors import LocalOutlierFactor

from lbl2vec.utils import _get_doc_label_similarities, centroid


class Lbl2Vec:
    '''
    Lbl2Vec

    Creates jointly embedded label, document and word vectors. Once the model is trained it contains document and label vectors.

    Parameters
    ----------

    keywords_list : iterable list of lists with descriptive keywords of type str.
            For each label at least one descriptive keyword has to be added as list of str.

    tagged_documents : iterable list of `gensim.models.doc2vec.TaggedDocument`_ elements, optional
            If you wish to train word and document vectors from scratch this parameter can not be None, whereas the :class:`doc2vec_model` parameter must be None. If you use a pretrained Doc2Vec model to load its learned word and document vectors this parameter has to be None.
            Input corpus, can be simply a list of elements, but for larger corpora, consider an iterable that streams
            the documents directly from disk/network.

    label_names : iterable list of str, optional
            Custom names can be defined for each label. Parameter values of label names and keywords of the same topic must have the same index. Default is to use generic label names.

    epochs : int, optional
            Number of iterations (epochs) over the corpus.

    vector_size : int, optional
            Dimensionality of the feature vectors.

    min_count : int, optional
            Ignores all words with total frequency lower than this.

    window : int, optional
            The maximum distance between the current and predicted word within a sentence.

    sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).

    negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.

    workers : int, optional
            The amount of worker threads to be used in training the model. Larger amount will lead to faster training.
            If set to -1, use all available worker threads of the machine.

    doc2vec_model : `gensim.models.doc2vec.Doc2Vec`_, optional
            If given a pretrained Doc2Vec model, Lbl2Vec uses its word and document vectors to compute the label vectors. If this parameter is defined, :class:`tagged_documents` has to be None.
            In order to get optimal Lbl2Vec results the given Doc2Vec model should be trained with the parameters "dbow_words=1" and "dm=0".

    num_docs : int, optional
            Maximum number of documents to calculate label embedding from. Default is all available documents.

    similarity_threshold : float, default=None
            Only documents with a higher similarity to the respective description keywords than this threshold are used to calculate the label embeddings.

    similarity_threshold_offset : float, default=0
            Sets similarity threshold to n-similarity_threshold_offset with n = (smiliarity of keyphrase_vector to most similar document_vector).

    min_num_docs : int, optional
            Minimum number of documents that are used to calculate the label embedding. Adds documents until requirement is fulfilled if simiarilty threshold is choosen too restrictive.
            This value should be chosen to be at least 1 in order to be able to calculate the label embedding.
            If this value is < 1 it can happen that no document is selected for label embedding calculation and therefore no label embedding is generated.

    clean_outliers : boolean, optional
        Whether to clean outlier candidate documents for label embedding calculation.
        Setting to False can shorten the training time.
        However, a loss of accuracy in the calculation of the label vectors may be possible.
        
    verbose : boolean, optional
        Whether to print status during training and prediction.
    '''

    def __init__(
            self,
            keywords_list: List[List[str]],
            tagged_documents: List[TaggedDocument] = None,
            label_names: List[str] = None,
            epochs: int = 10,
            vector_size: int = 300,
            min_count: int = 50,
            window: int = 15,
            sample: float = 1e-5,
            negative: int = 5,
            workers: int = -1,
            doc2vec_model: Doc2Vec = None,
            num_docs: int = None,
            # ToDo: check optimal similarity threshold value
            similarity_threshold: float = None,
            # ToDo: add validation checks for similarity_threshold_offset and add error/warning message in case the offset is larger than the highest label_vector <-> document_vector similarity
            similarity_threshold_offset: float = 0,
            min_num_docs: int = 1,
            clean_outliers: bool = False,
            verbose: bool = True):

        # validate keywords_list and label names
        if label_names is not None:
            if (not all(isinstance(i, str) for i in label_names)) or (
                    not isinstance(label_names, list)):
                raise ValueError('label_names has to be a list of str')

            if len(label_names) != len(keywords_list):
                raise ValueError(
                    'keywords_list and label_name have to be the same length')

        else:
            # create generic label names
            label_names = [('label_' + str(i))
                           for i in range(len(list(keywords_list)))]

        # validate keywords_list
        if (not isinstance(keywords_list, list)) or (not all(isinstance(i, list) for i in keywords_list)) or (
                not all(isinstance(i, str) for i in [item for sublist in keywords_list for item in sublist])):
            raise ValueError(
                'keywords_list has to be an iterable list of lists with descriptive keywords of type str')

        # ToDo (optional): auto convert keywords to lower case and remove empty keywords

        # init labels DataFrame
        self.labels = pd.DataFrame(list(zip(label_names, keywords_list)), columns=[
            'label_name', 'description_keywords'])

        # validate allowed tagged_documents/doc2vec_model parameter combination
        if not (tagged_documents is None) ^ (doc2vec_model is None):
            raise ValueError(
                'Either provide a pre-trained Doc2Vec model in the "doc2vec_model" paramater or provide tagged documents in the "tagged_documents" parameter to train a new Doc2Vec model. This is a logical XOR condition.')

        # validate tagged_documents
        if not (tagged_documents is None):
            if (not all((isinstance(i, TaggedDocument))
                        for i in tagged_documents)):
                raise ValueError(
                    'tagged_documents has to be an iterable list with elements of type ~gensim.models.doc2vec.TaggedDocument')

        # use all workers if parameter is < 1 or if workers parameter is bigger
        # than available number of cores
        if (workers < 1) or (workers > psutil.cpu_count(logical=True)):
            self.workers = psutil.cpu_count(logical=True)
        else:
            self.workers = workers

        # init parameters
        self.verbose = verbose
        self.clean_outliers = clean_outliers
        self.tagged_documents = tagged_documents
        self.epochs = epochs
        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.sample = sample
        self.negative = negative
        self.doc2vec_model = doc2vec_model
        self.max_num_docs = num_docs
        self.similarity_threshold = similarity_threshold
        self.similarity_threshold_offset = similarity_threshold_offset
        self.min_num_docs = min_num_docs

        # show or hide logging according to parameter setting
        self.logger = logging.getLogger('Lbl2Vec')
        self.logger.setLevel(logging.WARNING)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(sh)

        # show or hide logging according to parameter setting
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def fit(self):
        '''
        Trains the Lbl2Vec model which creates jointly embedded label, document and word vectors.

        '''

        # train new Doc2Vec model if no pre-trained model is given, else just
        # init pre-trained Doc2Vec model
        if self.doc2vec_model is None:

            # define Doc2Vec train parameters
            doc2vec_args = {"documents": self.tagged_documents,
                            "epochs": self.epochs,
                            "vector_size": self.vector_size,
                            "min_count": self.min_count,
                            "window": self.window,
                            "sample": self.sample,
                            "negative": self.negative,
                            "workers": self.workers,
                            "hs": 1,
                            "dm": 0,
                            "dbow_words": 1}

            self.logger.info('Train document and word embeddings')

            # train and init Doc2Vec model
            self.doc2vec_model = Doc2Vec(**doc2vec_args)

        else:
            self.logger.info('Load document and word embeddings')

            if not isinstance(self.doc2vec_model, type(Doc2Vec())):
                raise ValueError(
                    '"doc2vec_model" must be of type:', type(
                        Doc2Vec()))

        self.logger.info('Train label embeddings')

        # get doc keys and similarity scores of documents that are similar to
        # the description keywords
        self.labels[['doc_keys', 'doc_similarity_scores']] = self.labels['description_keywords'].apply(
            lambda row: self._get_similar_documents(
                self.doc2vec_model, row, num_docs=self.max_num_docs, similarity_threshold=self.similarity_threshold,
                min_num_docs=self.min_num_docs))

        # validate that documents to calculate label embeddings from are found
        # for all labels
        if len(self.labels[self.labels['doc_keys'].str.len() != 0]) != len(
                self.labels):
            raise ValueError(
                'The model could not find documents to calculate label embeddings from for each label. Either lower the "similarity_threshold" parameter or set the "max_num_docs" parameter > 0')

        # get document vectors from document keys
        self.labels['doc_vectors'] = self.labels['doc_keys'].apply(
            lambda row: self._get_doc_vectors(
                doc2vec_model=self.doc2vec_model, doc_keys=row))

        if self.clean_outliers:
            # clean document outlier with LOF
            self.labels['cleaned_doc_vectors'] = self.labels['doc_vectors'].apply(
                lambda row: self._remove_outlier_docs(document_vectors=row))

            # calculate centroid of document vectors as new label vector
            self.labels['label_vector_from_docs'] = self.labels['cleaned_doc_vectors'].apply(
                lambda row: self._get_centroid_from_vectors(row))
        else:
            # calculate centroid of document vectors as new label vector
            self.labels['label_vector_from_docs'] = self.labels['doc_vectors'].apply(
                lambda row: self._get_centroid_from_vectors(row))

    def predict_model_docs(
            self,
            doc_keys: Union[List[int], List[str]] = None,
            multiprocessing: bool = False) -> pd.DataFrame:
        '''
        Computes similarity scores of documents that are used to train the Lbl2Vec model to each label.

        Parameters
        ----------
        doc_keys : list of document keys, optional
            If None: return the similarity scores for all documents that are used to train the Lbl2Vec model.
            Else: only return the similarity scores of training documents with the given keys.

        multiprocessing : boolean, optional
            Whether to use the ray multiprocessing library during prediction.
            If True, ray uses all available workers for prediction.
            If False, just use a single core for prediction.

        Returns
        -------
        labeled_docs : `pandas.DataFrame`_ with first column of document keys, second column of most similar labels, third column of similarity scores of the document to the most similar label and the following columns with the respective labels and the similarity scores of the documents to the labels. The similarity scores consist of cosine similarities and therefore have a value range of [-1,1].
        '''

        # triggers a parameter validation
        if not hasattr(doc_keys, '__iter__') and (doc_keys is not None):
            raise ValueError(
                "Iterable over document keys expected."
            )

        # define column names for document keys, most similar label, highest
        # similarity score and prediction confidence
        doc_key_column = 'doc_key'
        most_similar_label_column = 'most_similar_label'
        highest_similarity_score_column = 'highest_similarity_score'
        prediction_confidence_column = 'prediction_confidence'

        self.logger.info('Get document embeddings from model')

        # get document keys and vectors from Doc2Vec model
        if doc_keys is None:

            keys_docVecs = [
                (self.doc2vec_model.dv.index_to_key[i],
                 self.doc2vec_model.dv.vectors[i]) for i in range(
                    len(
                        self.doc2vec_model.dv.vectors))]

        else:
            keys_docVecs = [(key, self.doc2vec_model.dv.get_vector(key))
                            for key in doc_keys]

        labeled_docs = pd.DataFrame(
            keys_docVecs, columns=[
                doc_key_column, 'doc_vec'])

        # calculate document vector <-> label vector similarities
        labeled_docs = self._get_document_label_similarities(labeled_docs=labeled_docs, doc_key_column=doc_key_column,
                                                             most_similar_label_column=most_similar_label_column,
                                                             highest_similarity_score_column=highest_similarity_score_column,
                                                             prediction_confidence_column=prediction_confidence_column,
                                                             multiprocessing=multiprocessing)
        return labeled_docs

    def predict_new_docs(
            self,
            tagged_docs: List[TaggedDocument],
            multiprocessing: bool = False) -> pd.DataFrame:
        '''
        Computes similarity scores of given new documents that are not used to train the Lbl2Vec model to each label.

        Parameters
        ----------
        tagged_docs : iterable list of `gensim.models.doc2vec.TaggedDocument`_ elements
            New documents that are not used to train the model.

        multiprocessing : boolean, optional
            Whether to use the ray multiprocessing library during prediction.
            If True, ray uses all available workers for prediction.
            If False, just use a single core for prediction.

        Returns
        -------
        labeled_docs : `pandas.DataFrame`_ with first column of document keys, second column of most similar labels, third column of similarity scores of the document to the most similar label and the following columns with the respective labels and the similarity scores of the documents to the labels. The similarity scores consist of cosine similarities and therefore have a value range of [-1,1].
        '''

        # validate tagged_documents
        if (not all((isinstance(i, TaggedDocument)) for i in tagged_docs)):
            raise ValueError(
                'tagged_docs has to be an iterable list with elements of type ~gensim.models.doc2vec.TaggedDocument')

        # define column names for document keys, most similar label and highest
        # similarity score
        doc_key_column = 'doc_key'
        most_similar_label_column = 'most_similar_label'
        highest_similarity_score_column = 'highest_similarity_score'
        prediction_confidence_column = 'prediction_confidence'

        # extract document keys and word lists from tagged_docs
        doc_keys, doc_words = map(
            list, zip(*[(doc[1][0], doc[0]) for doc in tagged_docs]))

        labeled_docs = pd.DataFrame(list(zip(doc_keys, doc_words)), columns=[
            doc_key_column, 'doc_word_tokens'])

        self.logger.info('Calculate document embeddings')

        # get document vectors of new documents
        if multiprocessing:
            try:
                if not ray.is_initialized():
                    # Start ray cluster
                    ray.init(num_cpus=psutil.cpu_count(logical=True), ignore_reinit_error=True, log_to_driver=False,
                             logging_level=logging.ERROR, configure_logging=False)
                    assert ray.is_initialized()
                # define distributed inference function
                doc2vec_model_id = ray.put(self.doc2vec_model)

                @ray.remote
                def infer_vector(doc2vec_model, doc_word_token):
                    return doc2vec_model.infer_vector(doc_words=doc_word_token)

                labeled_docs['doc_vec'] = ray.get(
                    [infer_vector.remote(doc2vec_model=doc2vec_model_id, doc_word_token=doc_word_token) for
                     doc_word_token in list(labeled_docs['doc_word_tokens'])])

            finally:
                ray.shutdown()
                assert not ray.is_initialized()

        else:
            labeled_docs['doc_vec'] = labeled_docs['doc_word_tokens'].apply(
                lambda row: self.doc2vec_model.infer_vector(doc_words=row))

        labeled_docs.drop(columns=['doc_word_tokens'], inplace=True)

        # calculate document vector <-> label vector similarities
        labeled_docs = self._get_document_label_similarities(labeled_docs=labeled_docs, doc_key_column=doc_key_column,
                                                             most_similar_label_column=most_similar_label_column,
                                                             highest_similarity_score_column=highest_similarity_score_column,
                                                             prediction_confidence_column=prediction_confidence_column,
                                                             multiprocessing=multiprocessing)

        return labeled_docs

    def add_lbl_thresholds(
            self,
            lbl_similarities_df: pd.DataFrame,
            lbl_thresholds: List[Tuple[str, float]]) -> pd.DataFrame:
        '''
        Adds threshold column with the threshold value of the most similar classification label.

        Parameters
        ----------
        lbl_similarities_df : `pandas.DataFrame`_ with first column of document keys, second column of most similar labels, third column of similarity scores of the document to the most similar label and the following columns with the respective labels and the similarity scores of the documents to the labels.
            This `pandas.DataFrame`_ type is returned by the :class:`predict_model_docs()` and :class:`predict_new_docs()` functions.

        lbl_thresholds : list of tuples
            First tuple element consists of the label name and the second tuple element of the threshold value.

        Returns
        -------
        lbl_similarities_df : `pandas.DataFrame`_ with first column of document keys, second column of most similar labels, third column of similarity scores of the document to the most similar label, fourth column of the label threshold values and the following columns with the respective labels and the similarity scores of the documents to the labels.
        '''
        # validate lbl_threshold
        if len(lbl_thresholds) != self.labels.shape[0]:
            raise ValueError(
                'Threshold list must be the same length as the keywords list')

        # validate pandas DataFrame by first three column names
        columns_names = [
            'doc_key',
            'most_similar_label',
            'highest_similarity_score']
        if not (lbl_similarities_df.columns[:3] == columns_names).all():
            raise ValueError(
                'The pandas DataFrame must have a first column of document keys, second column of most similar labels, third column of similarity scores of the document to the most similar label and the following columns with the respective labels and the similarity scores of the documents to the labels')

        lbl_similarities_df['lbl_threshold'] = lbl_similarities_df['most_similar_label'].apply(
            lambda x: lbl_thresholds[[element[0] for element in lbl_thresholds].index(x)][1])

        # reorder columns
        first_columns = [
            'doc_key',
            'most_similar_label',
            'highest_similarity_score',
            'lbl_threshold']
        following_columns = [
            e for e in lbl_similarities_df.columns.tolist() if e not in first_columns]
        column_order = first_columns + following_columns

        return lbl_similarities_df[column_order]

    def _get_document_label_similarities(self, labeled_docs: pd.DataFrame, doc_key_column: str,
                                         most_similar_label_column: str, highest_similarity_score_column: str,
                                         prediction_confidence_column: str, multiprocessing: bool) -> pd.DataFrame:
        '''
        Calculate the similarities of given document vectors to the label vectors
        Parameters
        ----------
        labeled_docs : pd.DataFrame
            DataFrame with document key and document vector column

        doc_key_column : str
            Column name for document keys

        most_similar_label_column : str
            Column name for  most similar label

        highest_similarity_score_column : str
            Column name  highest similarity score

        prediction_confidence_column : str
            Column name for prediction confidence

        multiprocessing : boolean, optional
            Whether to use the ray multiprocessing library during prediction.
            If True, ray uses all available workers for prediction.
            If False, just use a single core for prediction.

        Returns
        -------
        labeled_docs : `pandas.DataFrame`_ with first column of document keys, second column of most similar labels, third column of similarity scores of the document to the most similar label and the following columns with the respective labels and the similarity scores of the documents to the labels. The similarity scores consist of cosine similarities and therefore have a value range of [-1,1].
        '''

        self.logger.info('Calculate document<->label similarities')

        # get similarity scores of documents for each label
        if multiprocessing:
            try:
                if not ray.is_initialized():
                    # Start ray cluster
                    ray.init(num_cpus=psutil.cpu_count(logical=True), ignore_reinit_error=True, log_to_driver=False,
                             logging_level=logging.ERROR, configure_logging=False)
                    assert ray.is_initialized()
                label_name_id = ray.put(self.labels['label_name'])
                label_vector_form_docs_id = ray.put(self.labels['label_vector_from_docs'])
                distributed_get_doc_label_similarities = ray.remote(_get_doc_label_similarities)
                labeled_docs[list(self.labels['label_name'].values) + [prediction_confidence_column]] = ray.get([
                                                                                                                    distributed_get_doc_label_similarities.remote(
                                                                                                                        labels=label_name_id,
                                                                                                                        doc_vector=[
                                                                                                                            vector],
                                                                                                                        label_vectors=label_vector_form_docs_id)
                                                                                                                    for
                                                                                                                    vector
                                                                                                                    in
                                                                                                                    list(
                                                                                                                        labeled_docs[
                                                                                                                            'doc_vec'])])
            finally:
                ray.shutdown()
                assert not ray.is_initialized()

        else:
            labeled_docs[
                list(
                    self.labels['label_name'].values) +
                [prediction_confidence_column]] = labeled_docs['doc_vec'].apply(
                lambda row: _get_doc_label_similarities(
                    labels=self.labels['label_name'],
                    doc_vector=[row],
                    label_vectors=self.labels['label_vector_from_docs']))

        labeled_docs.drop(columns=['doc_vec'], inplace=True)

        # if the user wants to predict more than one label
        if (self.labels['label_name'].shape[0] > 1):

            # get most similar label for each row
            labeled_docs[most_similar_label_column] = labeled_docs.drop(
                [doc_key_column, prediction_confidence_column], axis=1).idxmax(axis=1)

            # get similarity score of most similar label for each row
            labeled_docs[highest_similarity_score_column] = labeled_docs.drop(
                [doc_key_column, most_similar_label_column, prediction_confidence_column], axis=1).max(axis=1)

            # reorder columns
            first_columns = [
                doc_key_column,
                most_similar_label_column,
                highest_similarity_score_column,
                prediction_confidence_column]
            following_columns = [
                e for e in labeled_docs.columns.tolist() if e not in first_columns]
            column_order = first_columns + following_columns
            # ToDo: drop prediction confidence column until a measurement is implemented that reports low confidence if all classes have a low similarity (unlike currently implemented softmax)
            labeled_docs = labeled_docs[column_order].drop([prediction_confidence_column], axis=1)

        # if the user only wants the similarities of one label it is not necessary to calculate the most similar label values
        else:
            # reorder columns
            first_columns = [doc_key_column, prediction_confidence_column]
            following_columns = [e for e in labeled_docs.columns.tolist() if e not in first_columns]
            column_order = [doc_key_column, following_columns[0]]
            labeled_docs = labeled_docs[column_order].rename(columns={doc_key_column: doc_key_column,
                                                                      following_columns[0]: str(
                                                                          following_columns[0] + '_similarity')})
        return labeled_docs

    def _get_similar_documents(self,
                               doc2vec_model: type(
                                   Doc2Vec()),
                               keywords: List[str],
                               num_docs: int,
                               similarity_threshold: float,
                               min_num_docs: int) -> pd.Series:
        '''
        Computes document keys and similarity scores of documents that are similar to given description keywords.

        Parameters
        ----------
        doc2vec_model : pretrained `gensim.models.doc2vec.Doc2Vec`_ embedding model

        keywords : list of strings
            Describing words of a label/classs which is searched for in the model.

        num_docs : int, optional
            Maximum number of documents to return. Default is all available documents.

        similarity_threshold : float, optional
            Only documents with a similarity score above this threshold are returned.

        min_num_docs : int, optional
            Minimum number of documents to return. Adds documents until requirement is fulfilled if similarity threshold is choosen to restrictive.
            This value should be chosen to be at least 1 in order to be able to calculate the label embedding.
            If this value is < 1 it can happen that no document is selected for label embedding calculation and therefore no label embedding is generated.

        Returns
        -------
        similar_docs : `pandas.Series`_ with columns of keys and similarity scores of documents which are similar to the given keywords.
        '''

        return_docs_keys = []
        return_docs_similarity_scores = []

        # if keywords list is not empty and all elements are strings
        if keywords and all(isinstance(word, str) for word in keywords):
            if num_docs is None:
                # set max_num_docs to maximum number of available documents in
                # doc2vec model
                num_docs = len(doc2vec_model.dv.index_to_key)

            # get topics which are similar to keywords in decreasing similarity
            # score order
            try:
                # remove keywords that are not contained in the trained Doc2Vec model
                cleaned_keywords_list = list(
                    set(keywords).intersection(
                        doc2vec_model.wv.index_to_key))
                removed_keywords_list = [
                    x for x in keywords if x not in set(cleaned_keywords_list)]
                if removed_keywords_list:
                    self.logger.warning(
                        "The following keywords from the 'keywords_list' are unknown to the Doc2Vec model and therefore not used to train the model: {}".format(
                            ' '.join(map(str, removed_keywords_list))))

                # get documents that are similar to all remaining keywords in
                # the list
                keyword_vectors = [doc2vec_model.wv[word]
                                   for word in cleaned_keywords_list]
                similar_docs = doc2vec_model.dv.most_similar(
                    positive=keyword_vectors, topn=num_docs)
            except KeyError as error:
                error.args = (
                                 error.args[
                                     0] + " in trained Doc2Vec model. Either replace the keyword from the 'keywords_list' parameter or train a new Doc2Vec model that knows the keyword.",) + error.args[
                                                                                                                                                                                              1:]
                raise

            doc_keys = [docs[0] for docs in similar_docs]
            doc_scores = [docs[1] for docs in similar_docs]

            if similarity_threshold is None:
                # set similarity threshold to n-similarity_threshold_offset with n = (smiliarity of keyphrase_vector to most similar document_vector)
                similarity_threshold = doc_scores[0] - self.similarity_threshold_offset

            # add number of min_num_docs documents if too few documents are
            # chosen by similarity threshold alone
            if min_num_docs is not None and doc_scores[min_num_docs] < similarity_threshold and len(
                    doc2vec_model.dv.index_to_key) >= min_num_docs:
                return_docs_similarity_scores = doc_scores[:min_num_docs]
                return_docs_keys = doc_keys[:min_num_docs]

            else:

                if similarity_threshold is not None:
                    # get only documents with similarity score > similarity_threshold
                    for i in range(num_docs):
                        if doc_scores[i] <= similarity_threshold:
                            break
                        return_docs_keys.append(doc_keys[i])
                        return_docs_similarity_scores.append(doc_scores[i])
                else:
                    return_docs_keys = doc_keys
                    return_docs_similarity_scores = doc_scores

            return pd.Series([return_docs_keys, return_docs_similarity_scores], index=[
                'doc_keys', 'doc_similarity_scores'])

        else:
            raise ValueError(
                'List of keywords must be not be empty and all elements must be of type "str"')

    def _get_doc_vectors(
            self,
            doc2vec_model: type(
                Doc2Vec()),
            doc_keys: Union[List[int], List[str]]) -> List[List[float]]:
        '''
        Computes a list of document vectors from a list of document keys.

        Parameters
        ----------
        doc2vec_model : pretrained `gensim.models.doc2vec.Doc2Vec`_ model

        doc_keys : list of document keys

        Returns
        -------
        doc_vecs : list of document vectors for each document key.
        '''
        return [doc2vec_model.dv.get_vector(key) for key in doc_keys]

    def _remove_outlier_docs(self, document_vectors: List[List[float]]) -> List[List[float]]:
        '''
        Removes outlier document vectors with local outlier factor.

        Parameters
        ----------
        document_vectors : list of document vectors

        Returns
        -------
        document_vectors : list of cleaned document vectors.
        '''
        if len(document_vectors) > 1:
            if len(document_vectors) < 20:
                n_neighbors = len(document_vectors)
            else:
                n_neighbors = 20
            lof_predictions = LocalOutlierFactor(
                n_neighbors=n_neighbors, n_jobs=self.workers).fit_predict(document_vectors)
            return [document_vectors[i] for i in range(
                len(lof_predictions)) if lof_predictions[i] == 1]
        else:
            return document_vectors

    def _get_centroid_from_vectors(self, vectors: List[List[float]]) -> np.array:
        '''
        Computes the centroid of given vectors.

        Parameters
        ----------
        vectors : list of vectors to calculate centroid from

        Returns
        -------
        centroid : `numpy.array`_ centroid of vectors.
        '''

        if len(vectors) > 0:
            vectors = [np.array(vector) for vector in vectors]
            return centroid(vectors)
        else:
            return []

    def save(self, filepath: str):
        '''
        Saves the Lbl2Vec model to disk.

        Parameters
        ----------
        filepath : str
            Path of file.
        '''
        if isinstance(filepath, str):
            pickle.dump(self, open(filepath + '.p', 'wb'))
        else:
            raise ValueError('filepath must be of type "str"')

    @classmethod
    def load(cls, filepath: str) -> object:
        '''
        Loads the Lbl2Vec model from disk.

        Parameters
        ----------
        filepath : str
            Path of file.

        Returns
        -------
        lbl2vec_model : Lbl2Vec model loaded from disk.
        '''
        if not isinstance(filepath, str):
            raise ValueError('filepath must be of type "str"')

        else:
            return pickle.load(open(filepath + '.p', 'rb'))
