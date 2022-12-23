"""
.. _np.array: https://numpy.org/doc/stable/reference/generated/numpy.array.html
.. _SentenceTransformer: https://www.sbert.net/docs/pretrained_models.html
"""

from typing import List, Union

import numpy as np
import pandas as pd
import syntok.segmenter as segmenter
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity
from syntok.tokenizer import Tokenizer
from transformers import AutoModel


def sentence_tokenize(text_document: str) -> List[str]:
    '''
    Converts a text document to a list of sentences.

    Parameters
    ----------
    text_document : str
            The text document to split into sentences
    Returns
    -------
    sentences : List[str]
        List of sentences
    '''
    # split token streams into list of tokens lists
    segments = segmenter.split(iter(Tokenizer().split(text_document)))

    # join token lists to sentence list of strings
    sentences = [(''.join([(token.spacing + token.value) for token in sentence]).strip()) for sentence in segments]

    return sentences


def split_long_document(text: str, max_text_length: int) -> List[str]:
    """
    Split single string in a paragraph (or list of sentences) with a maximum number of words.

    Parameters
    ----------
    text : str
        Text string that should be split.

    max_text_length : int
        Maximun number of words per paragraph.

    Returns
    -------
    splitted_document : List of text strings.
    """
    sentences = sentence_tokenize(text)

    text_list_len = len(sentences) - 1
    list_of_joined_srings_with_max_length = []
    one_string = ''
    for index, sentence in enumerate(sentences):
        if len(one_string.split()) + len(sentence.split()) < max_text_length:
            if one_string != '':
                one_string += ' ' + sentence

            else:
                one_string += sentence
            if index == text_list_len:
                list_of_joined_srings_with_max_length.append(one_string)

        # Substring too large, so add to the list and reset
        else:
            list_of_joined_srings_with_max_length.append(one_string)
            one_string = sentence
            if index == text_list_len:
                list_of_joined_srings_with_max_length.append(one_string)

    return list_of_joined_srings_with_max_length


def transformer_embedding(model: Union[SentenceTransformer, AutoModel], document: str, device: torch.device,
                          tokenizer=None) -> np.array:
    '''
    Uses a pretrained language transformer model to embed a text document as numeric vector.

    Parameters
    ----------
    model : Union[`SentenceTransformer`_, transformers.AutoModel]
            The pre-trained transformer model that is used to embed the text document

    document : str
            The text document that should be embedded

    device : torch.device
        Which torch.device to use for the computation

    tokenizer : transformers.AutoTokenizer, optional
        If the model is not of type `SentenceTransformer`_, this function needs the respective model tokenizer

    Returns
    -------
    doc_embedding : `np.array`_
        A vector representation embedding of the input document
    '''

    if type(model) == type(SentenceTransformer()):

        # split document in list of sentences
        sentences = split_long_document(text=document, max_text_length=int(model.max_seq_length * 0.5))

        # embed the sentences
        sentence_embeddings = model.encode(sentences=sentences, device=device)

    else:

        # triggers a parameter validation
        if tokenizer is None:
            raise ValueError(
                "For this transformer model type, you need to define the appropriate tokenizer."
            )

        # split document in list of sentences
        sentences = split_long_document(text=document, max_text_length=int(tokenizer.model_max_length * 0.5))

        # tokenize the sentences
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)

        # Get the embeddings
        with torch.no_grad():
            model.to(device)
            sentence_embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        # convert from embeddings from tensors to numpy arrays
        sentence_embeddings = sentence_embeddings.detach().cpu().numpy()

    # stack sentence embedding tensors and calculate mean as document embedding tensor
    doc_embedding = np.stack(sentence_embeddings).mean(axis=0)

    # return document embedding as numpy array
    return doc_embedding


def centroid(vectors: List[np.array]) -> np.array:
    '''
    Returns the centroid vector for a given list of vectors.

    Parameters
    ----------
    vectors : List[`np.array`_]
            The list of vectors to calculate the centroid from

    Returns
    -------
    centroid : `np.array`_
        The centroid vector
    '''
    # stack vectors and calculate mean as document embedding tensor
    centroid = np.stack(vectors).mean(axis=0)

    return centroid


def top_similar_vectors(key_vector: np.array, candidate_vectors: List[np.array]) -> List[tuple]:
    '''
     Calculates the cosines similarities of a given key vector to a list of candidate vectors.

     Parameters
     ----------
     key_vector : `np.array`_
             The key embedding vector

     candidate_vectors : List[`np.array`_]
             A list of candidate embedding vectors
     Returns
     -------
     top_results : List[tuples]
          A descending sorted of tuples of (cos_similarity, list_idx) by cosine similarities for each candidate vector in the list
     '''
    cos_scores = util.cos_sim(key_vector, np.asarray(candidate_vectors))[0]
    top_results = torch.topk(cos_scores, k=len(candidate_vectors))
    top_cos_scores = top_results[0].detach().cpu().numpy()
    top_indices = top_results[1].detach().cpu().numpy()

    return list(zip(top_cos_scores, top_indices))


def _get_doc_label_similarities(labels: List[str], doc_vector: List[float],
                                label_vectors: List[List[float]]) -> pd.Series:
    '''
    Computes the similarity scores of a given document vector to given label vectors.

    Parameters
    ----------
    labels : list of label names

    doc_vector : list
        One document vector

    label_vectors : list of label vectors

    Returns
    -------
    doc_label_similarities : `pandas.Series`_ with columns of labels and row of similarity scores.
    '''

    # returns a list of tuples where the first tuple element consists of a
    # similar label and the second tuple element consists of a similarity
    # score
    similarity_tuples = [
        (labels[i], cosine_similarity(
            doc_vector, [
                label_vectors[i]])[0][0]) for i in range(
            len(label_vectors))]

    # get prediction confidence
    confidence = _prediction_confidence(
        cos_similarities=[tpl[1] for tpl in similarity_tuples])
    prediction_confidence_column_name = 'prediction_confidence'

    similarity_tuples.append(
        (prediction_confidence_column_name, confidence))

    return pd.Series([tpl[1] for tpl in similarity_tuples], index=[
        tpl[0] for tpl in similarity_tuples])


def _prediction_confidence(cos_similarities: List[float]) -> float:
    '''
    Applies the softmax function with temperature scaling factor T to calibrate predictions for an optimized confidence probability of a labeling.

    Parameters
    ----------
    cos_similarities : list of floats with cosine similarities of a document to each label/class embedding

    Returns
    -------
    confidence: float
        Confidence of the prediction/labeling within the value range [0,1].
    '''
    # scaling parameter
    T = (1 / 20)

    # apply the temperature scaled softmax transformation to each cosine
    # similarity value and return the maximum of the transformed values
    return max(np.exp(np.array(cos_similarities) / T) /
               np.sum(np.exp(np.array(cos_similarities) / T)))
