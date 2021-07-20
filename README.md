Lbl2Vec
======= 

Lbl2Vec is an algorithm for **unsupervised document classification** and **unsupervised document retrieval.** It automatically generates jointly embedded label, document and word vectors and returns documents of topics modeled by manually predefined keywords. Once you train the Lbl2Vec model you can:

* Classify documents as related to one of the predefined topics.
* Get similarity scores for documents to each predefined topic.
* Get most similar predefined topic of documents.

Benefits
--------

1. No need to label the whole document dataset for classification.
2. No stop word lists required.
3. No need for stemming/lemmatization.
4. Works on short text.
5. Creates jointly embedded label, document, and word vectors.

How does it work?
-----------------

The key idea of the algorithm is that many semantically similar keywords can represent a topic. In the first step, the algorithm creates a joint embedding of document and word vectors. Once documents and words are embedded in a vector space, the goal of the algorithm is to learn label vectors from previously manually defined keywords representing a topic. Finally, the algorithm can predict the affiliation of documents to topics from *document vector <-> label vector* similarities. 

### The Algorithm
**0. Use the manually defined keywords for each topic of interest.**
>Domain knowledge is needed to define keywords that describe topics and are semantically similar to each other within the topics.

| Basketball     | Soccer        | Baseball   |
| :-------------:|:-------------:|:----------:|
| NBA            | FIFA          | MLB        |
| Basketball     | Soccer        | Baseball   |
| LeBron         | Messi         | Ruth       |
| ...            | ...           | ...        |


**1. Create jointly embedded document and word vectors using [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html "Gensim Doc2Vec").**
>Documents will be placed close to other similar documents and close to the most distinguishing words.

![](https://raw.githubusercontent.com/sebischair/Lbl2Vec/main/images/Doc2Vec_example.png)

**2. Find document vectors that are similar to the keyword vectors of each topic.**
>Each color represents a different topic described by the respective keywords. 

![](https://raw.githubusercontent.com/sebischair/Lbl2Vec/main/images/Document_assignment_example.png)

**3. Clean outlier document vectors for each topic.**
>Red documents are outlier vectors that are removed and do not get used for calculating the label vector. 

![](https://raw.githubusercontent.com/sebischair/Lbl2Vec/main/images/Outlier_cleaning_example.png)

**4. Compute the centroid of the outlier cleaned document vectors as label vector for each topic.**
>Points represent the label vectors of the respective topics. 

![](https://raw.githubusercontent.com/sebischair/Lbl2Vec/main/images/Label_vectors_example.png)

**5. Compute *label vector <-> document vector* similarities for each label vector and document vector in the dataset.**
>Documents are classified as topic with the highest *label vector <-> document vector* similarity.

![](https://raw.githubusercontent.com/sebischair/Lbl2Vec/main/images/Classification_example.png)

Installation
------------

```
pip install lbl2vec
```
    
Usage
-----
For detailed information visit the [Lbl2Vec API Guide](https://lbl2vec.readthedocs.io/en/latest/api.html#) and the [examples](https://github.com/sebischair/Lbl2Vec/tree/main/examples).

``` 
from lbl2vec import Lbl2Vec
```

### Learn new model from scratch
>Learns word vectors, document vectors and label vectors from scratch during Lbl2Vec model training.

``` 
# init model
model = Lbl2Vec(keywords_list=descriptive_keywords, tagged_documents=tagged_docs)
# train model
model.fit()
```
**Important parameters:**

* `keywords_list`: iterable list of lists with descriptive keywords of type str. For each label at least one descriptive keyword has to be added as list of str.
* `tagged_documents`: iterable list of [gensim.models.doc2vec.TaggedDocument](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.TaggedDocument) elements. If you wish to train a new Doc2Vec model this parameter can not be None, whereas the `doc2vec_model` parameter must be None. If you use a pretrained Doc2Vec model this parameter has to be None. Input corpus, can be simply a list of elements, but for larger corpora, consider an iterable that streams the documents directly from disk/network.

### Use word and document vectors from pretrained Doc2Vec model
>Uses word vectors and document vectors from a pretrained Doc2Vec model to learn label vectors during Lbl2Vec model training.

```
# init model
model = Lbl2Vec(keywords_list=descriptive_keywords, doc2vec_model=pretrained_d2v_model)
# train model
model.fit()
```

**Important parameters:**

* `keywords_list`: iterable list of lists with descriptive keywords of type str. For each label at least one descriptive keyword has to be added as list of str.
* `doc2vec_model`: pretrained [gensim.models.doc2vec.Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec) model. If given a pretrained Doc2Vec model, Lbl2Vec uses the pre-trained Doc2Vec model from this parameter. If this parameter is defined, `tagged_documents` parameter has to be None. In order to get optimal Lbl2Vec results the given Doc2Vec model should be trained with the parameters "dbow_words=1" and "dm=0".


### Predict label similarities for documents used for training
>Computes the similarity scores for each document vector stored in the model to each of the label vectors.

```
# get similarity scores from trained model
model.predict_model_docs()
```

**Important parameters:**

* `doc_keys`: list of document keys (optional). If None: return the similarity scores for all documents that are used to train the Lbl2Vec model. Else: only return the similarity scores of training documents with the given keys.

### Predict label similarities for new documents that are not used for training
>Computes the similarity scores for each given and previously unknown document vector to each of the label vectors from the model.

```
# get similarity scores for each new document from trained model
model.predict_new_docs(tagged_docs=tagged_docs)
```

**Important parameters:**

* `tagged_docs`: iterable list of [gensim.models.doc2vec.TaggedDocument](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.TaggedDocument) elements

### Save model to disk
``` 
model.save('model_name')
``` 

### Load model from disk
``` 
model = Lbl2Vec.load('model_name')
``` 
