from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tfidf_encode(docs, vocabulary):
    """
    Takes list of documents (requirement as sentence) and
    returns a matrix representation

    param docs: list of sentences
    param vocabulary: vocabulary to compute tfidf for (glossary terms)

    """

    list_of_lengths_of_terms = [len(term.split()) for term in vocabulary]

    ngram_range = (min(list_of_lengths_of_terms),
                   max(list_of_lengths_of_terms))

    tf_idf = TfidfVectorizer(lowercase=False, vocabulary=vocabulary, ngram_range=ngram_range)

    vectorizer = tf_idf.fit(docs)

    # to see what and how words are tokenized
    # print(vectorizer.vocabulary_)

    # transform/encode
    vector = vectorizer.transform(docs)

    # replace zeros with nan
    vector[vector == 0] = np.nan

    dense_vector = vector.T.todense()
    means = np.nanmean(dense_vector, axis=1)
    # convert to list of numbers
    means = [round(x[0], 3) for x in means.tolist()]

    # put in dictionary for indexing
    word_to_tfidf_mean = {x: y for x, y in zip(vectorizer.get_feature_names(), means)}

    return word_to_tfidf_mean

