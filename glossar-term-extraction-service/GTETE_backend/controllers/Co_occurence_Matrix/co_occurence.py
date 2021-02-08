from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re


class CoOccurrence:
    def __init__(self):
        self.model = CountVectorizer(binary=True, lowercase=False, token_pattern=u"(?u)\\b\\w+\\b")

    def transform_document(self, documents):
        vectorizer = self.model.fit(documents)
        features = vectorizer.get_feature_names()
        vector = vectorizer.transform(documents)
        co_occurrence_matrix = (vector.T * vector).todense()
        # map term to its vector in the matrix
        self.term_to_vector_map = {x: y for x, y in zip(features, co_occurrence_matrix.tolist())}

    def get_vector_from_term(self, term):
        """
        Gets glossary terms and returns mean vector.
        """
        vectors = []
        try:
            for word in re.split(' |-', term):
                vectors += [self.term_to_vector_map[word]]
        except KeyError:
            pass  # so far only numbers that get ignored
        return np.average(np.array([vectors]), axis=1)
