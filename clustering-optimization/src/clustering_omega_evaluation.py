# taking the implementation from https://github.com/isaranto/omega_index
from .Omega import Omega
import pandas as pd
from .glossary_terms_extractor import GlossaryTermsExtractor


class OmegaEvaluator:
    """
    Allows for omega index comparison for different clustering techniques.
    
    Takes
    ground truth path and gold glossary terms path
    
    Returns
    Omega index for two configurations (from already implemented Omega computation algorithm)
    """

    def __init__(self, ground_truth_path, gold_glossar_path, glossary_extractor):
        self.ground_truth = self.read_ground_truth(ground_truth_path)
        self.gold_glossary_terms = [x[:-1] for x in open(gold_glossar_path).readlines()]
        self._obtained = None
        self._recall = None
        self._omega_index_ = None
        self.glossary_extractor = glossary_extractor

    def clear_nans(self, d: dict):
        out = {}
        for a, b in d.items():
            out[a] = [x for x in b if x == x and x != 'nan']
        return out

    def read_ground_truth(self, file_: str):
        # clusters are organized into columns
        ground_truth = pd.read_csv(file_, sep=',', encoding='utf-8')
        # display(ground_truth)
        # [ {k:v for k,v in m.items() if ground_truth.notnull(v)} for m in df.to_dict(orient='rows')]
        to_dict_ = ground_truth.to_dict(orient='list')
        # clear NaNs
        return self.clear_nans(to_dict_)

    def evaluate(self, obtained, ground):
        omega = Omega(obtained, ground)
        # print('Obtained: \n')
        # display(obtained)
        # print('Ground: \n')
        # display(ground)
        return omega.omega_score

    @property
    def obtained(self):
        # The list of clustering output from a given clustering algorithm
        # self._obtained = self.clear_nans(fuzzy_clustering(lemmatized_to_original.keys())) 
        return self._obtained

    @obtained.setter
    def obtained(self, obt):
        self._obtained = obt

    @property
    def recall(self):
        count = 0
        print('Recall Calculation:\n ')
        print('Items in the gold list but not in glossary output:\n ')
        not_found = []
        for item in self.gold_glossary_terms:
            if item not in self.glossary_extractor.raw_glossary_terms:
                count += 1
                print(item)
                not_found.append(item)

        merged_reqs = ' \n '.join(self.glossary_extractor.requirements)
        # display(merged_reqs)
        print('\nItems not found above and not in requirements: \n')
        for g in not_found:
            # print(g)
            if g not in merged_reqs:
                print(g)

        self._recall = str(round((1 - count / len(self.gold_glossary_terms)) * 100, 2))
        # print('Recall = ' + self._recall + '%')
        return self._recall

    @property
    def omega_index_(self):
        self._omega_index_ = self.evaluate(self._obtained, self.ground_truth)
        return self._omega_index_


# gte = GlossaryTermsExtractor()
# oe = OmegaEvaluator(
#     'evaluation/GROUND_TRUTH_OPENCOSS.csv',
#     'evaluation/GOLD_GLOSSAR_OPENCOSS.txt',
#     gte
# )



