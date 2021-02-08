import itertools
import os

os.environ['SPACY_WARNING_IGNORE'] = 'W008'
import spacy
import math
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from ..BERT.BERT_sentence_embedding import AfosEmbedding
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cosine
import hunspell
import os.path
from charsplit import Splitter
import string

splitter = Splitter()

spellchecker = hunspell.HunSpell('/usr/share/hunspell/de_DE.dic',
                                 '/usr/share/hunspell/de_DE.aff')


class CreateClusters:

    def __init__(self, glossary_terms, maps, lang, documents):
        self.afos_embeddings = AfosEmbedding('GTETE_backend/models/BERT_models/bert-base-german-cased/')
        self.maps = maps
        self.documents = documents
        self.glossary_terms = glossary_terms
        if lang == 'DE':
            self.nlp = spacy.load('de_core_news_sm')
        elif lang == 'EN':
            self.nlp = spacy.load('en_core_news_sm')

    def recursive_split(self, glossary_term):
        ergebnis = []
        for word in glossary_term.split():
            if len(word) < 14:
                # check if it ends with s and remove it
                # if, after removing s, term is tagged as NOUN, then remove s
                # and return it as it is
                if word[-1] == 's' and len(word) > 1:
                    if self.nlp(word[:-1])[0].pos_ == 'NOUN':
                        ergebnis += [word[:-1]]
                    else:
                        ergebnis += [word]
                else:
                    ergebnis += [word]
            else:
                if splitter.split_compound(word)[0][0] == 0:
                    ergebnis += [splitter.split_compound(word)[0][1]]
                # elif splitter.split_compound(word)[0][0] < 0.85:
                elif splitter.split_compound(word)[0][0] < -0.6:
                    ergebnis += [word]
                else:
                    ergebnis += [item[:-1] if item[-1] == 's' and self.nlp(item)[0].pos_ != 'PROPN' else item for
                                 item in
                                 splitter.split_compound(word)[0][1:]]
        output = [glossary_term, ' '.join(ergebnis)]
        if output[1] == glossary_term:
            return glossary_term
        O = []
        for x in output[1].split():
            O.append(self.recursive_split(x))
        return ' '.join(O)

    def get_distance(self, glossary_terms, dist, type=None):
        distances = []
        for i in range(len(glossary_terms)):
            for j in range(i, len(glossary_terms)):
                if type == 'token':
                    sim = eval('textdistance.{}'.format(dist))(
                        glossary_terms[i].split(), glossary_terms[j].split()
                    )
                    if i != j:
                        if sim == 0:
                            sim = 0.0000001
                        dist_ = math.sqrt(-2 * math.log(sim))
                        distances.append([glossary_terms[i], glossary_terms[j], dist_])
                    else:
                        dist_ = 0
                    distances.append([glossary_terms[j], glossary_terms[i], dist_])
                elif type == 'sequence':
                    distances.append([glossary_terms[i], glossary_terms[j], 1 - eval('textdistance.{}'.format(dist))(
                        glossary_terms[i], glossary_terms[j]
                    )])
                else:  # character based
                    distances.append([glossary_terms[i], glossary_terms[j], eval('textdistance.{}'.format(dist))(
                        glossary_terms[i].replace(' ', ''), glossary_terms[j].replace(' ', '')
                    )])
        return distances

    distance_measures = ['hamming', 'levenshtein', 'jaro_winkler']
    similarity_measures = ['jaccard', 'sorensen']
    sequence_based = ['ratcliff_obershelp']

    def get_distances(self, glossary_terms, model=None):
        # get distances using sentence embeddings
        distances = []
        for i in range(len(glossary_terms)):
            for j in range(i, len(glossary_terms)):
                # Get the sentence vectors of both terms
                # if same terms, simply return 0
                if i != j:
                    if model == 'bert':
                        sentence_embeddings = self.afos_embeddings.sentence_embedding(glossary_terms[i],
                                                                                      glossary_terms[j])
                    elif model == 'co-occurrence':
                        sentence_embeddings = [self.maps[1][glossary_terms[i]], self.maps[1][glossary_terms[j]]]
                    # Compute cosine distance
                    inter_distance = cosine(sentence_embeddings[0], sentence_embeddings[1])
                    distances.append([glossary_terms[j], glossary_terms[i], inter_distance])
                else:
                    inter_distance = 0
                distances.append([glossary_terms[i], glossary_terms[j], inter_distance])
        return distances


    def split_terms(self):
        gesplitted_glossar = []
        for g in self.glossary_terms:
            gesplitted_glossar.append(self.recursive_split(g))
        return gesplitted_glossar

    def concat(self, setOfStrings):
        tmpString = " "
        for string in setOfStrings:
            tmpString = tmpString + " " + string
        return tmpString

    def create_clusters_(self, gesplitted_glossar):
        nounChunkAsSetToPlaintextMap = {}
        listOfNounSets = []
        for noun in gesplitted_glossar:
            nounSplittedAsSet = set(noun.split())
            listOfNounSets.append(nounSplittedAsSet)
            nounChunkAsSetToPlaintextMap[self.concat(nounSplittedAsSet)] = noun
        maxLength = max([len(nounSet) for nounSet in listOfNounSets])
        clusterList = []
        for i in range(maxLength, -1, -1):
            sublistforClusterSize_i = []
            listOfSetsWhichShouldBeRemoved = []
            for j in range(0, len(listOfNounSets)):
                if len(listOfNounSets) == 1:
                    sublistforClusterSize_i.append(listOfNounSets[j])
                for k in range(j + 1, len(listOfNounSets)):
                    # do not count article intersections
                    # go over all intersections and substract the number
                    # of articles found
                    number_of_articles = 0
                    if listOfNounSets[j] == listOfNounSets[k]:
                        continue
                    intersection = list(listOfNounSets[j].intersection(listOfNounSets[k]))
                    for u in range(len(intersection)):
                        doc_ = self.nlp(intersection[u])
                        if doc_[0].pos_ == 'ADP' or doc_[0].pos_ == 'DET':
                            number_of_articles += 1
                    if len(listOfNounSets[j].intersection(listOfNounSets[k])) - number_of_articles == i:
                        if len(listOfNounSets[j]) == i and len(listOfNounSets[k]) == i:
                            continue
                        else:
                            if listOfNounSets[j] not in sublistforClusterSize_i:
                                sublistforClusterSize_i.append(listOfNounSets[j])
                            if listOfNounSets[k] not in sublistforClusterSize_i:
                                sublistforClusterSize_i.append(listOfNounSets[k])
                            listOfSetsWhichShouldBeRemoved.append(listOfNounSets[j])
                            listOfSetsWhichShouldBeRemoved.append(listOfNounSets[k])
            clusterList.append(sublistforClusterSize_i)
            for entry in listOfSetsWhichShouldBeRemoved:
                if entry in listOfNounSets:
                    listOfNounSets.remove(entry)

        # format of clusterList is list(list(set())) respectively [[{}]]
        # list to store entries of the same cluster
        cluster_entries = []
        for entry in clusterList:
            tmp = []
            print("\n cluster " + str(maxLength) + " contains all nounchunks which have " + str(
                maxLength) + " word(s) in common: \n")
            if len(entry) == 0:
                print("---")
            for nounChunk in entry:
                print(nounChunkAsSetToPlaintextMap[self.concat(nounChunk)])
                tmp.append(nounChunkAsSetToPlaintextMap[self.concat(nounChunk)])
            cluster_entries.append(tmp)
            maxLength -= 1
        return cluster_entries

    def create_clusters(self, d, threshold=None):
        """
        Returns clusters dictionary using agglomerative hierarchical clustering.
        """
        distance_matrix = pd.DataFrame(data=d)
        dist_matrix = distance_matrix.pivot(0, 1, 2).fillna(0).values
        # model = AgglomerativeClustering(affinity='precomputed', n_clusters=None, linkage='complete',
        # distance_threshold=3).fit(dist_matrix)
        Z = linkage(squareform(dist_matrix), method='centroid')
        labels_ = fcluster(Z, threshold, criterion='distance')
        # model = SpectralClustering(n_clusters=n_clusters).fit_predict(dist_matrix)
        # labels_ = model.labels_
        # labels_ = model
        pairing = {a: b for a, b in zip(distance_matrix.pivot(0, 1, 2).fillna(0).index, labels_)}
        clusters = {a: [] for a in labels_}
        for label in range(1, max(labels_) + 1):
            for term in distance_matrix.pivot(0, 1, 2).fillna(0).index:
                if pairing[term] == label:
                    clusters[label] += [term]
        for key, value in zip(clusters.keys(), clusters.values()):
            print('--------subcluster %d-------' % key)
            for k in value:
                print(k)
        return clusters


    # semantic clustering of clusters
    def get_similarity(self, glossary_terms):
        # get similarities
        similarities = []
        # max length

        # nlp-doc the glossary terms and store
        nlp = spacy.load("de_core_news_sm")
        doc_glossary_terms = []
        for i in range(len(glossary_terms)):
            doc_glossary_terms.append(nlp(glossary_terms[i]))

        for i in range(len(glossary_terms)):
            for j in range(len(glossary_terms)):
                similarities.append(
                    [glossary_terms[i], glossary_terms[j], 1 - doc_glossary_terms[i].similarity(doc_glossary_terms[j])])

        return similarities

    def create_clusters_subclusters(self):
        """
        Generate main- and sub-clusters

        :return: cluster_index: the clustering label for each term
        """
        # maps[0]: term -> split form
        # maps[1]: split form -> vector
        # split_terms = self.maps[1].keys()
        split_terms = self.glossary_terms
        # lemmatization is irrelevant since now self.map_ contains split and lemmatized terms

        # map each split term to its non-split origin
        # this is useful in reconstructing documents with split terms

        # map each split term to its origin id
        # in the list of glossary terms
        # so that we can use that ID in the service table
        # term_to_id_map = {term: id_ for id_, term in enumerate(split_terms)}
        term_to_id_map = {}
        for id_, term in enumerate(split_terms):
            # a condition to make sure we include composite ids of split terms
            if term in term_to_id_map.keys():
                term_to_id_map[term] += [id_]
            else:
                term_to_id_map[term] = [id_]



        # Create clusters using new algorithm
        clusterList = self.create_clusters_(split_terms)
        # Display clusters
        i = len(clusterList) - 1
        print('\nCreating subclusters for each cluster obtained' + '\n')
        # dictionary that contains clusters and subclusters
        out = {}
        # indexes
        alpha = list(string.ascii_uppercase[:len(clusterList)])

        for i in range(len(clusterList) - 1):
            if clusterList[i] != []:

                # threshold for co-occurrence
                threshold = 0.09
                d = self.get_distances(clusterList[i], model='co-occurrence')
                subclusters = self.create_clusters(d, threshold=threshold)
                # remap subclusters to ids
                id_subclusters = {}
                keys = []
                for key, value in zip(subclusters.keys(), subclusters.values()):
                    for x in value:
                        keys += term_to_id_map[x]
                    id_subclusters[key] = keys
                    keys = []
                out[alpha[i]] = id_subclusters
        # convert out to cluster_index map
        out[alpha[-1]] = {0: list(itertools.chain.from_iterable([term_to_id_map[x] for x in clusterList[-1]]))}
        # out[alpha[-1]] = {0: [term_to_id_map[x] for x in clusterList[-1]]}
        cluster_index = {}
        for id_ in range(len(split_terms)):
            for alpha_ in out.keys():
                for number in out[alpha_].keys():
                    if id_ in out[alpha_][number]:
                        cluster_index[id_] = alpha_ + str(number)
        return cluster_index
