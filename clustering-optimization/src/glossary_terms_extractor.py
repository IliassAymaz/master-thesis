
import re
import spacy
from .recursive_split import recursive_split
from functools import lru_cache
import pprint
import os

pp = pprint.PrettyPrinter()

path_ = os.path.dirname(__file__)
# current = '../data/OPENCOSS_reqs.txt'
# current = '../data/arora_sentences.txt'
# DATA = open(os.path.join(path_, current), 'r').readlines()

# DATA = open('data/risiko_reqs.txt', 'r').readlines()

# REQUIREMENTS = [x.replace('\n', '') for x in DATA]

# GOLD_LIST = [x[:-1] for x in open('../evaluation/GOLD_GLOSSAR_OPENCOSS.txt', 'r', encoding='utf-8').readlines()]

# !python3.7 -m spacy download de_core_news_sm
nlp = spacy.load('de_core_news_sm')


class GlossaryTermsExtractor:

    def __init__(self, filter_terms=False):
        self._raw_glossary_terms = []
        self.glossary_term_to_ids = {}
        # only terms that also appear in the gold list
        self._filtered_glossary_terms = []
        self._glossary_terms = None
        self.filter_terms = filter_terms
        # gold list
        self._gold_list = None
        # reqs
        self._requirements = None
        self._original_to_split = {}
        self._split_to_lemmatized = {}
        self._original_to_lemmatized = {}
        self._lemmatized_to_original = {}
        self._split_to_ids = {}

    @property
    @lru_cache()
    def raw_glossary_terms(self):
        """
        Uses the REQUIREMENTS list (sentences in lines).
        Returns list of Noun Chunks along with trigrams; the list of raw glossary terms.
        The raw glossary terms are not lemmatized.
        """
        for i, r in enumerate(self._requirements):
            doc = nlp(r)
            for chunk in doc.noun_chunks:
                if chunk[0].is_stop:
                    glossar = ' '.join(re.findall(r'[\w-]+', ' '.join(chunk.text.split()[1:])))
                    self._raw_glossary_terms.append(glossar)
                else:
                    glossar = ' '.join(re.findall(r'[\w-]+', chunk.text))
                    self._raw_glossary_terms.append(glossar)
                self.store_glossar_in_id(glossar, i)

            text_and_pos_tuples = [[token.text, token.pos_] for token in doc]
            triGrams = [text_and_pos_tuples[i:i + 3] for i in range(len(text_and_pos_tuples) - 2)]
            for tri_gram in triGrams:
                if ((tri_gram[0][1] == "NOUN")
                        and (tri_gram[1][1] == "DET"
                             or tri_gram[1][1] == "ADP")
                        and (tri_gram[2][1] == "NOUN")):
                    glossar = ' '.join([tri_gram[i][0] for i in range(len(tri_gram))])
                    # glossar = lemmatizer.find_lemma(glossar, 'N')
                    # lemmatize
                    self._raw_glossary_terms.append(glossar)
                    # store id of sentence where the glossar appeared
                    # useful for cooccurrence matrix
                    self.store_glossar_in_id(glossar, i)

        # remove empty glossary terms
        self._raw_glossary_terms = [x for x in self._raw_glossary_terms if x]
        self._raw_glossary_terms = list(set(self._raw_glossary_terms))

        # remove glossary terms that start with a preposition (ADP)
        for term in self._raw_glossary_terms[:]:
            if nlp(term.split()[0])[0].pos_ == 'ADP':
                self._raw_glossary_terms.remove(term)

        return self._raw_glossary_terms

    def store_glossar_in_id(self, glossar, index_):
        """
        Stores glossary terms with their ids.
        """
        if glossar in self.glossary_term_to_ids:
            self.glossary_term_to_ids[glossar] += [index_]
        else:
            self.glossary_term_to_ids[glossar] = [index_]

    @property
    @lru_cache()
    def filtered_glossary_terms(self):
        self._filtered_glossary_terms = [term for term in self._raw_glossary_terms if term in self._gold_list]
        return self._filtered_glossary_terms

    @property
    def gold_list(self):
        return self._gold_list

    @gold_list.setter
    def gold_list(self, value):
        self._gold_list = value

    @property
    def requirements(self):
        return self._requirements

    @requirements.setter
    def requirements(self, value):
        self._requirements = value

    @property
    def glossary_terms(self):
        if self.filter_terms:
            self._glossary_terms = self.filtered_glossary_terms
        else:
            self._glossary_terms = self._raw_glossary_terms
        return self._glossary_terms

    @property
    @lru_cache()
    def original_to_split(self):
        for g in self.glossary_terms:
            self._original_to_split[g] = recursive_split(' '.join(re.findall(r'\w+', g)), nlp)
        return self._original_to_split

    @property
    @lru_cache()
    def split_to_lemmatized(self):
        for g in list(self.original_to_split.values()):
            self._split_to_lemmatized[g] = ' '.join([nlp(g)[i].lemma_ for i in range(len(nlp(g)))])
        return self._split_to_lemmatized

    @property
    @lru_cache()
    def original_to_lemmatized(self):
        for g in self.glossary_terms:
            self._original_to_lemmatized[g] = self.split_to_lemmatized[self.original_to_split[g]]
        return self._original_to_lemmatized

    @property
    @lru_cache()
    def lemmatized_to_original(self):
        # split to original map
        self._lemmatized_to_original = {b: a for a, b in self.original_to_lemmatized.items()}
        return self._lemmatized_to_original

    @property
    @lru_cache()
    def split_to_ids(self):
        for term in self.glossary_terms:
            self._split_to_ids[self.original_to_lemmatized[term]] = self.glossary_term_to_ids[term]
        return self._split_to_ids

    def fit(self):
        self.raw_glossary_terms


