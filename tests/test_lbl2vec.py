import re

import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing.preprocessing import strip_tags
from gensim.utils import simple_preprocess
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoModel

from lbl2vec import Lbl2Vec, Lbl2TransformerVec

# load data
train = fetch_20newsgroups(subset='train', shuffle=False)
test = fetch_20newsgroups(subset='test', shuffle=False)

# parse data to pandas DataFrames
newsgroup_test = pd.DataFrame({'article': test.data, 'class_index': test.target})
newsgroup_train = pd.DataFrame({'article': train.data, 'class_index': train.target})

# set labels
labels = pd.DataFrame(data={'class_index': [8, 9, 10, 11],
                            'class_name': ['rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt'],
                            'keywords': [['Bikes', 'Motorcycle'], ['Baseball'], ['Hockey'], ['Encryption', 'Privacy']]})


def test_lbl2vec():
    # convert description keywords to lowercase
    labels['keywords'] = labels['keywords'].apply(
        lambda description_keywords: [keyword.lower() for keyword in description_keywords])

    # get number of keywords for each class
    labels['number_of_keywords'] = labels['keywords'].apply(lambda row: len(row))

    # doc: document text string
    # returns tokenized document
    # strip_tags removes meta tags from the text
    # simple preprocess converts a document into a list of lowercase tokens, ignoring tokens that are too short or too long
    # simple preprocess also removes numerical values as well as punktuation characters
    def tokenize(doc):
        return simple_preprocess(strip_tags(doc), deacc=True, min_len=2, max_len=15)

    # add data set type column
    newsgroup_train['data_set_type'] = 'train'
    newsgroup_test['data_set_type'] = 'test'

    # concat train and test data
    newsgroup_full_corpus = pd.concat([newsgroup_train, newsgroup_test]).reset_index(drop=True)

    # reduce dataset to only articles that belong to classes where we defined our keywords
    newsgroup_full_corpus = newsgroup_full_corpus[
        newsgroup_full_corpus['class_index'].isin(list(labels['class_index']))]

    # tokenize and tag documents for Lbl2Vec training
    newsgroup_full_corpus['tagged_docs'] = newsgroup_full_corpus.apply(
        lambda row: TaggedDocument(tokenize(row['article']), [str(row.name)]), axis=1)

    # add doc_key column
    newsgroup_full_corpus['doc_key'] = newsgroup_full_corpus.index.astype(str)

    # add class_name column
    newsgroup_full_corpus = newsgroup_full_corpus.merge(labels, left_on='class_index', right_on='class_index',
                                                        how='left')

    # init model with parameters
    Lbl2Vec_model = Lbl2Vec(keywords_list=list(labels.keywords), tagged_documents=newsgroup_full_corpus['tagged_docs'][
        newsgroup_full_corpus['data_set_type'] == 'train'], label_names=list(labels.class_name),
                            similarity_threshold=0.43, min_num_docs=100, epochs=3)

    # train model
    Lbl2Vec_model.fit()

    # predict similarity scores
    model_docs_lbl_similarities = Lbl2Vec_model.predict_model_docs()

    assert model_docs_lbl_similarities.shape == (2390, 7)


def test_lbltransformer2vec_sbert_cpu():

    labels['class_name'] = labels['keywords'].apply(lambda row: ' and '.join(row))

    # add data set type column
    newsgroup_train['data_set_type'] = 'train'
    newsgroup_test['data_set_type'] = 'test'

    # concat train and test data
    newsgroup_full_corpus = pd.concat([newsgroup_train, newsgroup_test]).reset_index(drop=True)

    # reduce dataset to only articles that belong to classes where we defined our keywords
    newsgroup_full_corpus = newsgroup_full_corpus[
        newsgroup_full_corpus['class_index'].isin(list(labels['class_index']))]

    # remove non-alphanumerical chars and unify whitespace
    newsgroup_full_corpus['article'] = newsgroup_full_corpus['article'].apply(
        lambda row: re.sub(' +', ' ', re.sub(r'[^A-Za-z0-9 ,.!?@-]+', '', row)))

    # add class_name column
    newsgroup_full_corpus = newsgroup_full_corpus.merge(labels, left_on='class_index', right_on='class_index',
                                                        how='left')
    # add doc_key column
    newsgroup_full_corpus['doc_key'] = list(range(0, newsgroup_full_corpus.shape[0], 1))

    # init model
    lbl2vec_model = Lbl2TransformerVec(keywords_list=list(labels['keywords']),
                                       documents=newsgroup_full_corpus['article'],
                                       label_names=list(labels['class_name']), min_num_docs=100)

    # train model
    lbl2vec_model.fit()

    # predict similarity scores
    model_docs_lbl_similarities = lbl2vec_model.predict_model_docs()

    assert model_docs_lbl_similarities.shape == (3980, 7)


def test_lbltransformer2vec_simcse_cpu():
    labels['class_name'] = labels['keywords'].apply(lambda row: ' and '.join(row))

    # add data set type column
    newsgroup_train['data_set_type'] = 'train'
    newsgroup_test['data_set_type'] = 'test'

    # concat train and test data
    newsgroup_full_corpus = pd.concat([newsgroup_train, newsgroup_test]).reset_index(drop=True)

    # reduce dataset to only articles that belong to classes where we defined our keywords
    newsgroup_full_corpus = newsgroup_full_corpus[
        newsgroup_full_corpus['class_index'].isin(list(labels['class_index']))]

    # remove non-alphanumerical chars and unify whitespace
    newsgroup_full_corpus['article'] = newsgroup_full_corpus['article'].apply(
        lambda row: re.sub(' +', ' ', re.sub(r'[^A-Za-z0-9 ,.!?@-]+', '', row)))

    # add class_name column
    newsgroup_full_corpus = newsgroup_full_corpus.merge(labels, left_on='class_index', right_on='class_index',
                                                        how='left')
    # add doc_key column
    newsgroup_full_corpus['doc_key'] = list(range(0, newsgroup_full_corpus.shape[0], 1))

    # select transformer model
    transformer_model_name = 'princeton-nlp/sup-simcse-roberta-large'
    transformer_model = AutoModel.from_pretrained(transformer_model_name)

    # init model with parameters
    lbl2vec_model = Lbl2TransformerVec(transformer_model=transformer_model, keywords_list=list(labels['keywords']),
                                       documents=newsgroup_full_corpus['article'],
                                       label_names=list(labels['class_name']), min_num_docs=100, workers=1)

    # train model
    lbl2vec_model.fit()

    # predict similarity scores
    model_docs_lbl_similarities = lbl2vec_model.predict_model_docs()

    assert model_docs_lbl_similarities.shape == (3980, 7)
