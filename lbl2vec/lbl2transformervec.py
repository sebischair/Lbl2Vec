"""
.. _pandas.DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
.. _pandas.Series: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
.. _np.array: https://numpy.org/doc/stable/reference/generated/numpy.array.html
.. _SentenceTransformer: https://www.sbert.net/docs/pretrained_models.html
"""

import logging
import os
from typing import List, Union

import pandas as pd
import psutil
import ray
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from transformers import logging as transformers_logs

pd.options.mode.chained_assignment = None
import numpy as np
from lbl2vec.lbl2vec import Lbl2Vec
from lbl2vec.utils import transformer_embedding, centroid, top_similar_vectors


class Lbl2TransformerVec(Lbl2Vec):
    '''
    Lbl2TransformerVec

    Creates jointly embedded label and document vectors with transformer language models. Once the model is trained it contains document and label vectors.

    Parameters
    ----------

    keywords_list : iterable list of lists with descriptive keywords of type str
            For each label at least one descriptive keyword has to be added as list of str.

    documents : iterable list of strings
            Iterable list of text documents

    transformer_model : Union[`SentenceTransformer`_, transformers.AutoModel], default=SentenceTransformer('all-MiniLM-L6-v2')
            Transformer model used to embed the labels, documents and keywords.

    label_names : iterable list of str, default=None
            Custom names can be defined for each label. Parameter values of label names and keywords of the same topic must have the same index. Default is to use generic label names.

    similarity_threshold : float, default=None
            Only documents with a higher similarity to the respective description keywords than this threshold are used to calculate the label embeddings.

    similarity_threshold_offset : float, default=0
            Sets similarity threshold to n-similarity_threshold_offset with n = (smiliarity of keyphrase_vector to most similar document_vector).

    min_num_docs : int, default=1
            Minimum number of documents that are used to calculate the label embedding. Adds documents until requirement is fulfilled if simiarilty threshold is choosen too restrictive.
            This value should be chosen to be at least 1 in order to be able to calculate the label embedding.
            If this value is < 1 it can happen that no document is selected for label embedding calculation and therefore no label embedding is generated.

    max_num_docs : int, default=None
        Maximum number of documents to calculate label embedding from. Default is all available documents.

    clean_outliers : boolean, default=False
        Whether to clean outlier candidate documents for label embedding calculation.
        Setting to False can shorten the training time.
        However, a loss of accuracy in the calculation of the label vectors may be possible.

    workers : int, default=-1
        Use these many worker threads to train the model (=faster training with multicore machines).
        Setting this parameter to -1 uses all available cpu cores.
        If using GPU, this parameter is ignored.

    device : torch.device, default=torch.device('cpu')
        Specify the device that should be used for training the model.
        Default is to use the CPU device.
        To use CPU, set device to 'torch.device('cpu')'.
        To use GPU, you can e.g. specify 'torch.device('cuda:0')'.

    verbose : boolean, default=True
        Whether to print status during training and prediction.
    '''

    def __init__(
            self,
            keywords_list: List[List[str]],
            documents: List[str],
            transformer_model: Union[SentenceTransformer, AutoModel] = SentenceTransformer('all-MiniLM-L6-v2'),
            label_names: List[str] = None,
            # ToDo: check optimal similarity threshold value/offset
            similarity_threshold: float = None,
            # ToDo: add validation checks for similarity_threshold_offset and add error/warning message in case the offset is larger than the highest label_vector <-> document_vector similarity
            similarity_threshold_offset: float = 0,
            min_num_docs: int = 1,
            max_num_docs: int = None,
            clean_outliers: bool = False,
            workers: int = -1,
            device: torch.device = torch.device('cpu'),
            verbose: bool = True, ):

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

        # init labels DataFrame
        self.labels = pd.DataFrame(list(zip(label_names, keywords_list)), columns=[
            'label_name', 'description_keywords'])

        # triggers a parameter validation
        if isinstance(documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        # triggers a parameter validation
        if similarity_threshold is not None:
            if not isinstance(similarity_threshold, float) or not (-1 <= similarity_threshold <= 1):
                raise ValueError(
                    "similarity_threshold value has to be a float value betweeen -1 and 1"
                )

        # triggers a parameter validation
        if type(device) != type(torch.device('cpu')):
            raise ValueError(
                "Device needs to be of type torch.device. To use CPU, set device to 'torch.device('cpu')'. To use GPU, you can e.g. specify 'torch.device('cuda:0')'."
            )

        # triggers a parameter validation
        if not hasattr(documents, '__iter__'):
            raise ValueError(
                "Iterable over raw text documents expected."
            )

        # triggers a parameter validation
        if not isinstance(min_num_docs, int):
            raise ValueError(
                "min_num_docs must be of type int."
            )
        # triggers a parameter validation
        if (min_num_docs < 1):
            raise ValueError(
                "min_num_docs must be > 0 and < max_num_docs"
            )

        # triggers a parameter validation
        if max_num_docs is not None:
            # triggers a parameter validation
            if not isinstance(max_num_docs, int):
                raise ValueError(
                    "max_num_docs must be of type int."
                )
            # triggers a parameter validation
            if max_num_docs <= min_num_docs:
                raise ValueError(
                    "max_num_docs must be > min_num_docs"
                )

        # triggers a parameter validation
        if (workers < -1) or (workers > psutil.cpu_count(logical=True)) or (workers == 0):
            raise ValueError(
                "'workers' parameter value cannot be 0 and must be between -1 and " + str(
                    psutil.cpu_count(logical=True))
            )

        # init parameters
        self.device = device

        if self.device.type != 'cpu':
            self.workers = 1
        else:
            self.workers = workers

        self.transformer_model = transformer_model
        self.documents = pd.DataFrame(list(documents), columns=['doc'])
        self.verbose = verbose
        self.clean_outliers = clean_outliers
        self.max_num_docs = max_num_docs
        self.similarity_threshold = similarity_threshold
        self.min_num_docs = min_num_docs
        self.similarity_threshold_offset = similarity_threshold_offset

        if type(self.transformer_model) == type(SentenceTransformer()):
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model.name_or_path)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # show or hide logging according to parameter setting
        self.logger = logging.getLogger('Lbl2TransformerVec')
        self.logger.setLevel(logging.WARNING)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(sh)

        transformers_logs.set_verbosity_error()

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)

    def fit(self):
        '''
        Trains the Lbl2TransformerVec model which creates jointly embedded label and document vectors.
        '''

        # embed keywords with chosen transformer model
        self.logger.info('Compute keyword embeddings')
        self.labels['keyword_vectors'] = self.labels['description_keywords'].apply(
            lambda row: [transformer_embedding(model=self.transformer_model, document=keyword, device=self.device,
                                               tokenizer=self.tokenizer) for
                         keyword in row])

        # calculate mean keyword embedding
        self.labels['mean_keyword_vector'] = self.labels['keyword_vectors'].apply(lambda row: centroid(vectors=row))

        self.logger.info('Compute document embeddings')
        # embed document vectors on multiple cpus simultaneously
        if self.workers != 1:
            try:
                # Start ray cluster
                if self.workers == -1:
                    if not ray.is_initialized():
                        # Start ray cluster
                        ray.init(num_cpus=psutil.cpu_count(logical=True), ignore_reinit_error=True, log_to_driver=False,
                                 logging_level=logging.ERROR, configure_logging=False)
                        assert ray.is_initialized()
                else:
                    if not ray.is_initialized():
                        # Start ray cluster
                        ray.init(num_cpus=self.workers, ignore_reinit_error=True, log_to_driver=False,
                                 logging_level=logging.ERROR, configure_logging=False)
                        assert ray.is_initialized()

                transformer_model_id = ray.put(self.transformer_model)
                distributed_transformer_embedding = ray.remote(transformer_embedding)
                # embed documents with chosen transformer model
                self.documents['doc_vec'] = ray.get(
                    [distributed_transformer_embedding.remote(model=transformer_model_id, document=doc,
                                                              device=torch.device('cpu'), tokenizer=self.tokenizer) for
                     doc in
                     list(self.documents['doc'])])
            finally:
                ray.shutdown()
                assert not ray.is_initialized()
        # use single core only to embed document vectors
        else:
            self.documents['doc_vec'] = self.documents['doc'].apply(
                lambda row: transformer_embedding(model=self.transformer_model, document=row, device=self.device,
                                                  tokenizer=self.tokenizer))

        self.logger.info('Train label embeddings')
        # get document vectors that are similar to the mean keyword embedding
        self.labels['doc_vectors'] = self.labels['mean_keyword_vector'].apply(
            lambda row: self._get_similar_documents(keyphrase_vector=row,
                                                    document_vectors=list(self.documents['doc_vec']),
                                                    similarity_threshold=self.similarity_threshold,
                                                    max_num_docs=self.max_num_docs, min_num_docs=self.min_num_docs))

        if self.clean_outliers:
            # clean document outlier with LOF
            self.labels['cleaned_doc_vectors'] = self.labels['doc_vectors'].apply(lambda row: centroid(vectors=row))

            # calculate centroid of document vectors as new label vector
            self.labels['label_vector_from_docs'] = self.labels['cleaned_doc_vectors'].apply(
                lambda row: centroid(vectors=row))
        else:
            # calculate centroid of document vectors as new label vector
            self.labels['label_vector_from_docs'] = self.labels['doc_vectors'].apply(lambda row: centroid(vectors=row))

    def predict_model_docs(self, doc_idxs: List[int] = None) -> pd.DataFrame:
        '''
        Computes similarity scores of documents that are used to train the Lbl2TransformerVec model to each label.

        Parameters
        ----------
        doc_idxs : list of document indices, default=None
            If None: return the similarity scores for all documents that are used to train the Lbl2TransformerVec model.
            Else: only return the similarity scores of training documents with the given indices.

        Returns
        -------
        labeled_docs : `pandas.DataFrame`_ with first column of document texts, second column of most similar labels, third column of similarity scores of the document to the most similar label and the following columns with the respective labels and the similarity scores of the documents to the labels. The similarity scores consist of cosine similarities and therefore have a value range of [-1,1].
        '''
        # triggers a parameter validation
        if isinstance(doc_idxs, int) and (doc_idxs is not None):
            raise ValueError(
                "Iterable over integer indices expected. Only single integer received"
            )

        # triggers a parameter validation
        if isinstance(doc_idxs, str) and (doc_idxs is not None):
            raise ValueError(
                "Iterable over integer indices expected. String received"
            )

        # triggers a parameter validation
        if not hasattr(doc_idxs, '__iter__') and (doc_idxs is not None):
            raise ValueError(
                "Iterable over integer indices expected."
            )

        # define column names for document keys, most similar label, highest
        # similarity score and prediction confidence
        doc_key_column = 'doc_key'
        most_similar_label_column = 'most_similar_label'
        highest_similarity_score_column = 'highest_similarity_score'

        self.logger.info('Get document embeddings from model')

        # get document keys and vectors from Doc2Vec model
        if doc_idxs is not None:
            labeled_docs = self.documents.iloc[doc_idxs]
        else:
            labeled_docs = self.documents

        self.logger.info('Calculate document<->label similarities')
        # calculate document vector <-> label vector similarities
        labeled_docs = self._get_document_label_similarities(labeled_docs=labeled_docs, doc_key_column=doc_key_column,
                                                             most_similar_label_column=most_similar_label_column,
                                                             highest_similarity_score_column=highest_similarity_score_column)

        return labeled_docs

    def predict_new_docs(
            self,
            documents: List[str],
            workers: int = -1,
            device: torch.device = torch.device('cpu')) -> pd.DataFrame:
        '''
        Computes similarity scores of given new documents that are not used to train the Lbl2TransformerVec model to each label.

        Parameters
        ----------
        documents : iterable list of strings
            New documents that are not used to train the model.

        workers : int, default=-1
            Use these many worker threads to train the model (=faster training with multicore machines).
            Setting this parameter to -1 uses all available cpu cores.
            If using GPU, this parameter is ignored.

        device : torch.device, default=torch.device('cpu')
            Specify the device that should be used for training the model.
            Default is to use the CPU device.
            To use CPU, set device to 'torch.device('cpu')'.
            To use GPU, you can e.g. specify 'torch.device('cuda:0')'.

        Returns
        -------
        labeled_docs : `pandas.DataFrame`_ with first column of document texts, second column of most similar labels, third column of similarity scores of the document to the most similar label and the following columns with the respective labels and the similarity scores of the documents to the labels. The similarity scores consist of cosine similarities and therefore have a value range of [-1,1].
        '''

        # triggers a parameter validation
        if isinstance(documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        # triggers a parameter validation
        if not hasattr(documents, '__iter__'):
            raise ValueError(
                "Iterable over raw text documents expected."
            )

        # triggers a parameter validation
        if type(device) != type(torch.device('cpu')):
            raise ValueError(
                "Device needs to be of type torch.device. To use CPU, set device to 'torch.device('cpu')'. To use GPU, you can e.g. specify 'torch.device('cuda:0')'."
            )

        # triggers a parameter validation
        if (workers < -1) or (workers > psutil.cpu_count(logical=True)) or (workers == 0):
            raise ValueError(
                "'workers' parameter value cannot be 0 and must be between -1 and " + str(
                    psutil.cpu_count(logical=True))
            )


        if device.type != 'cpu':
            workers = 1

        # define column names for document keys, most similar label and highest
        # similarity score
        doc_key_column = 'doc_key'
        most_similar_label_column = 'most_similar_label'
        highest_similarity_score_column = 'highest_similarity_score'
        prediction_confidence_column = 'prediction_confidence'

        self.logger.info('Compute document embeddings')

        # convert document list to DataFrame
        labeled_docs = pd.DataFrame(list(documents), columns=['doc'])

        # embed documents with transformer model
        if workers != 1:
            try:
                # Start ray cluster
                if workers == -1:
                    ray.init(num_cpus=psutil.cpu_count(logical=True), ignore_reinit_error=True)
                else:
                    ray.init(num_cpus=workers, ignore_reinit_error=True)

                transformer_model_id = ray.put(self.transformer_model)
                distributed_transformer_embedding = ray.remote(transformer_embedding)
                # embed documents with chosen transformer model
                labeled_docs['doc_vec'] = ray.get(
                    [distributed_transformer_embedding.remote(model=transformer_model_id, document=doc,
                                                              device=torch.device('cpu'), tokenizer=self.tokenizer) for
                     doc in
                     list(labeled_docs['doc'])])
            finally:
                ray.shutdown()
        # use single core only to embed document vectors
        else:
            labeled_docs['doc_vec'] = labeled_docs['doc'].apply(
                lambda row: transformer_embedding(model=self.transformer_model, document=row, device=self.device,
                                                  tokenizer=self.tokenizer))

        # calculate document vector <-> label vector similarities
        labeled_docs = self._get_document_label_similarities(labeled_docs=labeled_docs, doc_key_column=doc_key_column,
                                                             most_similar_label_column=most_similar_label_column,
                                                             highest_similarity_score_column=highest_similarity_score_column)

        return labeled_docs

    def _get_similar_documents(self, keyphrase_vector: np.array, document_vectors: List[np.array],
                               similarity_threshold: float, max_num_docs: int, min_num_docs: int) -> List[np.array]:
        '''
        Returns the most similar document vectors to a keyphrase vector from a list of document vectors.

        Parameters
        ----------
        keyphrase_vector : `np.array`_
                The keyhprase embedding vector

        document_vectors : List[`np.array`_]
                A list of document embedding vectors

        similarity_threshold : float
            Only documents with a higher similarity to the respective description keyphrase vector than this threshold are returned

        max_num_docs : int
            Maximum number of similar documents to return

        min_num_docs : int
            Minimum number of documents to return

        Returns
        -------
        candidate_doc_vecs : List[`np.array`_]
            A list of top k most similar document vectors to the keyphrase vector
        '''

        if max_num_docs is None:
            # set max_num_docs to maximum number of available documents
            max_num_docs = len(document_vectors)
        document_vectors = document_vectors[:max_num_docs]

        top_results = top_similar_vectors(key_vector=keyphrase_vector, candidate_vectors=document_vectors)
        top_cos_scores = [element[0] for element in top_results]
        top_indices = [element[1] for element in top_results]
        top_results_df = pd.DataFrame([top_cos_scores, top_indices]).transpose().rename(
            columns={0: "cos_scores", 1: "doc_indices"})
        top_results_df['doc_indices'] = top_results_df['doc_indices'].astype(int)

        # only consider max_num_docs documents
        top_results_df = top_results_df.head(max_num_docs)

        if similarity_threshold is None:
            # set similarity threshold to n-similarity_threshold_offset with n = (smiliarity of keyphrase_vector to most similar document_vector)
            similarity_threshold = top_results_df['cos_scores'].iloc[0] - self.similarity_threshold_offset

        # add number of min_num_docs documents if too few documents are
        # chosen by similarity threshold alone
        if (min_num_docs is not None) and (
                top_results_df[top_results_df['cos_scores'] > similarity_threshold].shape[0] < min_num_docs):
            top_results_df = top_results_df.head(min_num_docs)

        else:
            # get only documents with similarity score > similarity_threshold
            top_results_df = top_results_df[top_results_df['cos_scores'] > similarity_threshold]

        similar_document_vectors = [document_vectors[i] for i in list(top_results_df['doc_indices'])]

        return similar_document_vectors

    def _get_document_label_similarities(self, labeled_docs: pd.DataFrame, doc_key_column: str,
                                         most_similar_label_column: str,
                                         highest_similarity_score_column: str) -> pd.DataFrame:
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

        Returns
        -------
        label_similarities_df : `pandas.DataFrame`_ with first column of document keys, second column of most similar labels, third column of similarity scores of the document to the most similar label and the following columns with the respective labels and the similarity scores of the documents to the labels. The similarity scores consist of cosine similarities and therefore have a value range of [-1,1].
        '''
        doc_keys = list(labeled_docs.index)

        label_similarities = []
        for label_vector in list(self.labels['label_vector_from_docs']):
            similarities = top_similar_vectors(key_vector=label_vector, candidate_vectors=list(labeled_docs['doc_vec']))
            similarities.sort(key=lambda x: x[1])
            similarities = [elem[0] for elem in similarities]
            label_similarities.append(similarities)

        label_similarities_df = pd.DataFrame(label_similarities).transpose()
        label_similarities_df.columns = list(self.labels['label_name'])
        label_similarities_df[doc_key_column] = doc_keys

        # get most similar label for each row
        label_similarities_df[most_similar_label_column] = label_similarities_df.drop(
            [doc_key_column], axis=1).idxmax(axis=1)

        # get similarity score of most similar label for each row
        label_similarities_df[highest_similarity_score_column] = label_similarities_df.drop(
            [doc_key_column, most_similar_label_column], axis=1).max(axis=1)

        # reorder columns
        first_columns = [
            doc_key_column,
            most_similar_label_column,
            highest_similarity_score_column]
        following_columns = [
            e for e in label_similarities_df.columns.tolist() if e not in first_columns]
        column_order = first_columns + following_columns
        label_similarities_df = label_similarities_df[column_order]

        return label_similarities_df
