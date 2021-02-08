import pprint

import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform, cosine
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import CountVectorizer
import skfuzzy as fz
from .recursive_split import recursive_split
import spacy
import numpy as np
import re

pp = pprint.PrettyPrinter()
nlp = spacy.load('de_core_news_sm')


class EvaluationSwitch:
    """

    

    Tested algorithms:

    AHC, Fuzzy C-Means, K-Means

    """

    def __init__(self, glossary_terms_extractor, threshold=None, number_of_clusters=None, plot_ahc=False):
        self._distance_matrix = []
        self.glossary_terms_extractor = glossary_terms_extractor
        self._distaptcher = {}
        self.threshold = threshold
        self.number_of_clusters = number_of_clusters
        self.plot_ahc = plot_ahc

    '''def switch(self, clustering_algorithm, embedding):
        return eval('self.{0}(*self.arguments({0})'.format(clustering_algorithm))


    def arguments(self, embedding):
        # map of parameters of different algorithms
        return {'agglomerative_hierarchical_clustering': [0.9, self.distance_matrix(vectors)]}'''

    @property
    def distaptcher(self):
        return {
            'bert': self.bert_embeddings,
            'co-occurrence': self.co_occurrence_embeddings,
            'word2vec': self.word2vec_embeddings,
            'agglomerative-hierarchical-clustering': self.agglomerative_hierarchical_clustering,
            'c-means': self.c_means,
            'keyword-clustering': self.keyword_clustering
        }

    def switch(self, clustering_alg, embedding):
        self.clustering_alg = clustering_alg
        self.embedding = embedding

    def fit(self):
        import pickle
        # pickle up vectors for fast scripting

        if self.embedding:
            try:
                with open('cache/vectors_OPENCOSS_{}.pickle'.format(self.embedding), 'rb') as f:
                    self.vectors = pickle.load(f)
                    # pp.pprint(self.vectors)
            except:
                self.vectors = self.distaptcher[self.embedding]()
                with open('cache/vectors_OPENCOSS_{}.pickle'.format(self.embedding), 'wb') as f:
                    pickle.dump(self.vectors, f, protocol=2)

            self.obtained = self.distaptcher[self.clustering_alg](self.vectors)
        else:
            # no embeddings are needed, therefore no argument is needed for now
            self.obtained = self.distaptcher[self.clustering_alg]()
        # pp.pprint(self.vectors)
        # pp.pprint(self.obtained)

    def agglomerative_hierarchical_clustering(self, vectors):
        """
        Input: 
        
            d: distance matrix of terms in the form [term1, term2, numerical distance]
            threshold: 
        Returns clusters dictionary using agglomerative hierarchical clustering.
        """
        # translate vectors to distances
        # threshold defined internally, for evaluation purposes
        if self.threshold:
            threshold = self.threshold
        else:
            threshold = 0.115
        d = []
        terms = list(vectors.keys())
        # create distance matrix from vectors
        for i in range(len(terms)):
            for j in range(i, len(terms)):
                d.append([terms[i], terms[j], cosine(vectors[terms[i]], vectors[terms[j]])])
                if i != j:
                    d.append([terms[j], terms[i], cosine(vectors[terms[j]], vectors[terms[i]])])
        # pp.pprint(d)
        distance_matrix = pd.DataFrame(data=d)
        dist_matrix = distance_matrix.pivot(0, 1, 2).fillna(0).values
        Z = linkage(squareform(dist_matrix), method='centroid')
        labels_ = fcluster(Z, threshold, criterion='distance')
        pairing = {a: b for a, b in zip(distance_matrix.pivot(0, 1, 2).fillna(0).index, labels_)}
        clusters = {a: [] for a in labels_}
        for label in range(1, max(labels_) + 1):
            for term in distance_matrix.pivot(0, 1, 2).fillna(0).index:
                if pairing[term] == label:
                    clusters[label] += [term]

        # For command line display of clusters
        '''for key, value in clusters.items():
            # print('--------subcluster %d-------' % key)
            print(f'{key}: {value}')'''

        # For plotting
        if self.plot_ahc:
            params = {'axes.labelsize': 16,
                      'axes.titlesize': 16,
                      'ytick.labelsize': 6}
            plt.rcParams.update(params)
            plt.title('Hierarchical Clustering Dendrogram')
            data_ = dendrogram(Z, labels=list(pairing.keys()))  # , leaf_rotation=90)
            for i, d in zip(data_['icoord'], data_['dcoord']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                plt.plot(x, y, 'ro')
                plt.annotate('%.3g' % y, (x, y), xytext=(0, -7), textcoords='offset points',
                             va='top', ha='center')

            # threshold
            threshold_ = [threshold] * len(data_)
            plt.axhline(y=threshold, color='r', linestyle='dashed')
            plt.xticks(rotation=45, ha="right", fontsize=12)
            # plot_dendrogram(Z)# labels=model.labels_)
            plt.show()

        return self.process_clusters_output(clusters)

    def k_means(self, *args):
        pass

    def c_means(self, vectors):
        # membership threshold
        threshold = 0.5
        to_be_stacked = ()
        for x in list(vectors.values()):
            to_be_stacked += (np.transpose(x),)
        all_data = np.hstack(to_be_stacked)
        # obtain clusters
        center, u, u0, *_, fpc = fz.cluster.cmeans(
            all_data, self.number_of_clusters, 1.3, error=0.005, maxiter=1000, init=None
        )
        # store fpcs and u's for visualization and selection
        clusters = {u: [] for u in range(self.number_of_clusters)}
        glossary_terms = list(vectors.keys())
        for j in range(len(glossary_terms)):
            for i in range(self.number_of_clusters):
                if u[i, j] > (1 / self.number_of_clusters) * 1:  # I am adjusting the fuzziness here
                    clusters[i] += [glossary_terms[j]]

        '''for j in range(len(glossary_terms)):
            if any(u[:, j] > threshold):
                for i in range(self.number_of_clusters):
                    if u[i, j] > threshold:
                        # append term to clusters that are higher than threshold only
                        clusters[i] += [glossary_terms[j]]
            else:
                # append term to all clusters
                for i in range(self.number_of_clusters):
                    clusters[i] += [glossary_terms[j]]'''
        return self.process_clusters_output(clusters)

    def keyword_clustering(self):


        keywords = []
        list_of_glossary_terms = list(self.glossary_terms_extractor.lemmatized_to_original.keys())
        for glossar in list_of_glossary_terms:
            keywords += glossar.split()
        # to uniques
        keywords = list(set(keywords))
        # eliminate stops from keywords
        keywords = [x for x in keywords if x not
                    in nlp.Defaults.stop_words]

        # remove empty entries
        list_of_glossary_terms = [x for x in list_of_glossary_terms if x]

        # map term to term as set for intersection operation
        term_to_term_as_set = {}
        for term in list_of_glossary_terms:
            # term_to_term_as_set[term] = set(term.split())
            term_to_term_as_set[term] = set(re.findall(r'\w+', term))
        for term in keywords:
            term_to_term_as_set[term] = set(re.findall(r'\w+', term))

        # sort subcluster by length of terms
        sorted_subcluster = sorted(list_of_glossary_terms, key=lambda x: len(x.split()))
        # display(sorted_subcluster)
        # and link every singular word to other terms with intersections
        generated_clusters = {}
        # label = 0

        # new approach
        for i in range(len(keywords)):
            a = term_to_term_as_set[keywords[i]]
            tmp = []
            for j in range(len(sorted_subcluster)):
                b = term_to_term_as_set[sorted_subcluster[j]]
                # if a != b:
                intersection_ = a.intersection(b)
                if intersection_:
                    tmp += [self.glossary_terms_extractor.lemmatized_to_original[sorted_subcluster[j]]]
            if len(tmp) > 1:
                generated_clusters[keywords[i]] = list(set(tmp))

        # add elements not belonging to any cluster to a separate cluster
        not_found = []
        for to_be_found in [self.glossary_terms_extractor.lemmatized_to_original[x] for x in list_of_glossary_terms]:
            found = False
            for list_ in generated_clusters.values():
                if to_be_found not in list_:
                    continue
                else:
                    found = True
                    break
            if found:
                continue
            else:
                not_found.append(to_be_found)
        generated_clusters['der Rest'] = not_found

        # remove clusters with one item
        d = generated_clusters.copy()
        for j in d.keys():
            if len(generated_clusters[j]) < 2:
                print(generated_clusters[j])
                del generated_clusters[j]
        del d

        # remove empty clusters
        generated_clusters = {a: b for a, b in generated_clusters.items() if b}

        # remove "long" (irrelvant) terms (questionable)
        for _, cluster in generated_clusters.items():
            for term in cluster[:]:
                if len(term.split()) > 5:
                    cluster.remove(term)

        # fill list with nan to have similar size
        # max len of cluster items

        return generated_clusters

    def co_occurrence_embeddings(self) -> dict:

        model = CountVectorizer(binary=True, lowercase=False, token_pattern=u"(?u)\\b\\w+\\b")

        def transform_document(documents):
            vectorizer = model.fit(documents)
            features = vectorizer.get_feature_names()
            vector = vectorizer.transform(documents)
            co_occurrence_matrix = (vector.T * vector).todense()
            return features, co_occurrence_matrix

        def document_to_matrix(features, co_occurrence_matrix):
            # map term to its vector in the matrix
            return {x: y for x, y in zip(features, co_occurrence_matrix.tolist())}

        def get_vector_from_term(term: str, co_occurrence_matrix):
            """
            Gets glossary terms and their matrix and returns mean vector.
            """
            vectors = []
            try:
                for word in re.split(' |-', term):
                    vectors += [co_occurrence_matrix[word]]
            except KeyError:
                print(term)
                pass  # so far only numbers that get ignored
            return np.average(np.array([vectors]), axis=1)

        def glossary_terms_to_sentences():
            # create a list of "sentences" that contain only noun chunks
            concatenated_ids = []
            for element in self.glossary_terms_extractor.split_to_ids.values():
                concatenated_ids += element
            max_id = max(concatenated_ids)
            sentences = [''] * (max_id + 1)
            for glossar in self.glossary_terms_extractor.split_to_ids.items():
                for x in glossar[1]:
                    sentences[x] += glossar[0] + ' '
            return sentences

        outs = transform_document(glossary_terms_to_sentences())
        term_to_vector_map = document_to_matrix(*outs)

        # setting new attributes to glossary_terms_extractor object
        self.glossary_terms_extractor.split_to_vector = {}
        '''for key in list(self.glossary_terms_extractor.split_to_ids.keys()):
            self.glossary_terms_extractor.split_to_vector[key] = get_vector_from_term(key, term_to_vector_map)'''

        self.glossary_terms_extractor.original_to_vector = {}
        for lemma, original in self.glossary_terms_extractor.lemmatized_to_original.items():
            self.glossary_terms_extractor.original_to_vector[original] = get_vector_from_term(lemma, term_to_vector_map)

        return self.glossary_terms_extractor.original_to_vector

    def bert_embeddings(self) -> dict:

        def sentence_embedding_pytorch(text):
            MODEL_ = "bert-base-german-cased"  # to be downloaded from the web!
            tokenizer = BertTokenizer.from_pretrained(MODEL_)
            model = BertModel.from_pretrained(MODEL_, output_hidden_states=True)
            model.eval()

            # Feed each sentence into a separate model
            encoded_text = tokenizer.encode_plus(
                text,
                add_special_tokens=True
            )
            tokens_tensor = torch.tensor([encoded_text['input_ids']])
            segments_tensor = torch.tensor([encoded_text['token_type_ids']])

            # model
            # Run the text through BERT, and collect all of the hidden states produced
            # from all 12 layers.
            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensor)
                hidden_states = outputs[2]

            # Concatenate the last 4 layers
            # sentences_cat is now a tensor[len_sentence x 3072 (=768x4)]
            # sentences_cat = torch.cat(
            #     (hidden_states[-4][0], hidden_states[-3][0], hidden_states[-2][0], hidden_states[-1][0]), dim=1)

            # alternative: consider only the layer -2 (in an attempt to reduce dimensionality for C-Means)
            sentences_cat = hidden_states[-2][0]

            # Get mean of the vectors to obtain the sentence embedding
            # convert to a proper np.array
            return np.array([np.array(torch.mean(sentences_cat, dim=0))])

        glossary_terms_dict = self.glossary_terms_extractor.lemmatized_to_original
        out = {}
        i = 0
        for lemma, original_form in glossary_terms_dict.items():
            out[original_form] = sentence_embedding_pytorch(lemma)
            # out[original_form] = np.average(list(sentence_embedding(lemma).values())[1:-1], axis=0)
            i += 1
            print('Generating vectors ... '
                  + str(round(
                eval('100*i/len(glossary_terms_dict)'), 2)
            ) + ' %', end='\r')
        return out

    def word2vec_embeddings(self, *args):
        pass

    def fasttext(self, *args):
        pass

    def distance_matrix(self, vectors_map: dict):
        """
        Gets vectors and returns a distance matrix
        """
        terms = list(vectors_map.keys())
        for i in len(range(terms)):
            for j in len(range(terms)):
                self._distance_matrix.append([terms[i], terms[j], cosine(vectors_map[terms[i]], vectors_map[terms[j]])])
        return self._distance_matrix

    def process_clusters_output(self, clusters_dict: dict):
        cluster0 = []
        seen = []
        for key, cluster in clusters_dict.copy().items():
            if cluster in seen:
                del clusters_dict[key]
            else:
                seen.append(cluster)
                if len(cluster) == 1:
                    cluster0 += cluster
                    del clusters_dict[key]
                if len(cluster) == 0:
                    del clusters_dict[key]
        clusters_dict['cluster0'] = cluster0

        return clusters_dict


# running example
# from glossary_terms_extractor import GlossaryTermsExtractor

'''
gte = GlossaryTermsExtractor(filter_terms=False)
gte.raw_glossary_terms
ec = EvaluationSwitch(gte)
ec.switch('agglomerative-hierarchical-clustering', 'bert')
ec.fit()
pp.pprint(ec.obtained)
'''
