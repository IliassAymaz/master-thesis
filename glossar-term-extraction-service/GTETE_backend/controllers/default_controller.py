import connexion

import re
import spacy
import pandas as pd
from wordfreq import word_frequency
from .utils.generateNegativeLists import group_negative_list
from .Clustering.clusters import CreateClusters
import json
import copy
import os
from .TF_IDF.tf_idf import tfidf_encode
from .Co_occurence_Matrix.co_occurence import CoOccurrence
from charsplit import Splitter
from math import log
import hunspell

splitter = Splitter()

from GTETE_backend.models.software_requirement import SoftwareRequirement  # noqa: E501
from GTETE_backend.models.statistics_table import StatisticsTable  # noqa: E501


def get_statistics(requirements_list):  # noqa: E501
    """Return statistic table with absolute and relative term frequencies for all verbs

    The API takes a list of requirements in JSON format and returns a table with absolute and relative term frequencies, statistical ratios and a list of requirements IDs where the verb appears.  # noqa: E501

    :param requirements_list: List of Software requirements that shall be analyzed
    :type requirements_list: list | bytes

    :rtype: List[StatisticsTable]
    """

    def create_clusters(dataframe, maps, lang, sentences):
        """
        Generates clusters from the set of glossary terms.
        Clusters are generated through comparison of common terms
        clusters are again clustered into subclusters.

        Current clustering algorithm on clusters: Agglomerative Hierarchical Clustering

        """
        # Get split terms
        dataframe['split_form'] = [maps[0][term] for term in list(dataframe['term'])]
        glossary_terms_split = list(dataframe['split_form'])
        clusters_creator = CreateClusters(glossary_terms_split, maps, lang, sentences)
        cluster_index_map = clusters_creator.create_clusters_subclusters()
        for id_, clusterIndex in zip(cluster_index_map.keys(), cluster_index_map.values()):
            dataframe.at[id_, 'Cluster_Ergebnis'] = clusterIndex
        return dataframe

    class Pipeline:
        """
        Token based pipeline.

        Parameters:
            hunspell_str (str): string of used hunspell dictionary.
            spacy_str (str): string of spacy neural network model.
            req_dataset (
        """

        def __init__(self,
                     hunspell_str,
                     spacy_str,
                     req_dataset,
                     co_occurrence_object,
                     lang=None,
                     csv_template=None):
            # csv template : True if requirements are formatted as columns in a csv file
            # or if german sentences follow the template (... mussen/sollen/bieten/sein .. object .. verb)
            self.hunspell_str = hunspell_str
            self.spacy_str = spacy_str
            self.req_dataset = req_dataset
            self.lang = lang
            self.co_occurrence_object = co_occurrence_object
            self.spellchecker = hunspell.HunSpell('/usr/share/hunspell/{}.dic'.format(self.hunspell_str),
                                                  '/usr/share/hunspell/{}.aff'.format(self.hunspell_str))
            self.nlp = spacy.load(self.spacy_str)
            if self.lang == 'DE':
                self.reqs = copy.deepcopy(self.req_dataset)
                if csv_template:
                    # if the requirements follow the template (... mussen/sollen/bieten/sein .. object .. verb)
                    for _ in range(len(self.reqs)):
                        self.match = re.match(r'(.+(muss|müssen|soll)) (.+(sein|bieten), )?(.+)( zu)? (.+n)\.',
                                              self.reqs[_]['text'])
                        self.reqs[_]['text'] = str(self.match.group(5)) + ' ' + str(self.match.group(7))
                        # remove the 'zu' at the end of the matches
                        if self.reqs[_]['text'].endswith('zu'):
                            self.reqs[_]['text'] = self.reqs[_]['text'][:-2]
            elif self.lang == 'EN':
                if csv_template:
                    # if object is separated within the csv
                    english_requirements = pd.read_csv(self.req_dataset, sep=';')
                    self.reqs = english_requirements['object']
                else:
                    self.reqs = copy.deepcopy(self.req_dataset)

            self.docs = [self.nlp(self.reqs[x]['text']) for x in range(len(self.reqs))]
            # summon the negative dictionary of POS lists
            self.pos_dict = group_negative_list(self.lang, "GTETE_backend/models/Black List %s.txt" % self.lang)
            # self.main_pattern = r'[^\W_\-\d]+'  # match words, except numbers and _
            # self.main_pattern = r'[^\W_\d]+'
            # self.main_pattern = r'[\w-]+'
            self.main_pattern = r'\w+[-\\.]?\w+'
            self.total_number_of_words = 0
            for i in range(len(self.reqs)):
                self.total_number_of_words += len(re.findall(self.main_pattern, self.reqs[i]['text']))
            # load stop words and adjust the german ones
            self.stops = list(self.nlp.Defaults.stop_words)
            if self.lang == 'DE':
                # + common 100 german words
                self.stops += open('GTETE_backend/models/Häufigste_Wörter_im_Deutschen.txt', 'r').readlines()
            # Afokorpus statistics
            self.statistics = json.loads(open("GTETE_backend/data/statistics.json", 'r').read())
            self.total_number_of_words_Afokorpus = 1
            self.matched_reqs = []
            for i in range(len(self.reqs)):
                self.reqs[i]['text'] = re.sub(r'[\d]+', '##', self.reqs[i]['text'])
                matcher = re.findall(self.main_pattern, self.reqs[i]['text'])
                self.matched_reqs.append(matcher)

        def recursive_split(self, glossary_term):
            ergebnis = []
            for word in glossary_term.split():
                if word[-1] == '-':
                    word = word[:-1]
                elif word[0] == '-':
                    word = word[1:]

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
                    # if word ends with -, remove the - first since it causes problems with splitting
                    if splitter.split_compound(word)[0][0] == 0:
                        ergebnis += [splitter.split_compound(word)[0][1]]
                    elif splitter.split_compound(word)[0][0] < -0.6:
                        ergebnis += [word]
                    else:
                        ergebnis += [
                            item[:-1] if item[-1] == 's' and self.nlp(item)[0].pos_ != 'PROPN' else item for
                            item in
                            splitter.split_compound(word)[0][1:]]
            output = [glossary_term, ' '.join(ergebnis)]
            if output[1] == glossary_term:
                return glossary_term
            O = []
            for x in output[1].split():
                O.append(self.recursive_split(x))
            return ' '.join(O)

        def run_pipeline_tokens(self, get_verbs=None, on_process_verbs=None, get_adjectives=None):
            """
            IWP: Individual Words Pipeline.
            Parameters:
                get_verbs (bool): True if verbs should be detected.
                on_process_verbs (bool): True if process verbs should be detected.
                get_adjectives (bool): True if adjectives should be detected.

            Returns:
                final (list): List of tuples of id, glossary candidate.
                verbs (list): List of tuples of id, verbs.
                adjectives (list): List of tuples of id, adjectives.
                first_candidates (list): list of tuples of id, abbreviations
            """

            # list of all words, repetition included
            words_list = []
            for i in range(len(self.matched_reqs)):
                # populate words[] with all words
                if self.matched_reqs[i] != []:
                    if not on_process_verbs:
                        for match in self.matched_reqs[i]:
                            words_list.append([i + 1, match])
                    else:
                        # get process (last word) verb only
                        words_list.append([i + 1, self.matched_reqs[i][-1]])
            # get the words from the tuple
            words = []
            for x in range(len(words_list)):
                words.append(words_list[x][1])

            # Get the abbreviations out of the pipeline
            # remove spelling mistakes
            # only misspelled words should stay in words[]
            first_candidates = []
            f_ = open('GTETE_backend/models/gängige Abkürzungen.txt', 'r').readlines()
            # remove return to line
            f_ = [item[:-1].lower() for item in f_]
            for x in range(len(words_list)):
                if not self.spellchecker.spell(words_list[x][1]) and words_list[x][1].lower() not in f_:
                    first_candidates.append((words_list[x][0], words_list[x][1]))
            first_candidates_einzel = [item[1] for item in first_candidates]
            # remove first_candiates from words[]
            words_list = [item for item in words_list if item[1] not in first_candidates_einzel]

            # lemmatize
            all_words = ' '.join([item[1] for item in words_list])
            doc = self.nlp(all_words)
            lemmatized_words = []
            for token in doc:
                lemmatized_words += [token.lemma_]
            lemmas = []
            for i in range(len(words_list)):
                lemmas.append([words_list[i][0], lemmatized_words[i]])
            words_list = lemmas

            # save detected verbs
            if get_verbs:
                # optional feature that gets the verbs, including with conjunction "und"
                verbs = []
                # read negative verbs list
                filter_ = open('GTETE_backend/models/negative Liste verbs.txt', 'r').readlines()
                filter_ = [item[:-1] for item in filter_]
                filter_ = filter_ + list(self.pos_dict["VERB"])
                for i in range(len(self.docs)):
                    for j in range(len(self.docs[i])):
                        if self.docs[i][j].pos_ == 'VERB' and self.docs[i][j].lemma_ not in filter_:
                            verbs.append((i + 1, self.docs[i][j].lemma_))
                # append "und" verbs
                for u in range(len(self.docs)):
                    i = 0
                    verbs_mit_und = []
                    while i + 2 <= len(self.docs[u]):
                        if self.lang == 'DE':
                            if self.docs[u][i].pos_ == 'VERB' and self.docs[u][i + 1].text == 'und' and self.docs[u][
                                i + 2].pos_ == 'VERB' \
                                    and self.docs[u][i].lemma_ not in filter_ \
                                    and self.docs[u][i + 2].lemma_ not in filter_:
                                verbs.append((u, ' '.join(
                                    [self.docs[u][i].lemma_, self.docs[u][i + 1].lemma_, self.docs[u][i + 2].lemma_])))
                                verbs_mit_und.append((u, ' '.join(
                                    [self.docs[u][i].lemma_, self.docs[u][i + 1].lemma_, self.docs[u][i + 2].lemma_])))
                        i += 1
                        if i == len(self.docs[u]):
                            break
                    verbs = list(set(verbs))
                return verbs, first_candidates

            if get_adjectives:
                # optionally extract adjectives
                neg_adj = list(self.pos_dict["ADJ"])
                neg_adj += ["gem", "muss"]
                adjectives = []
                for doc in self.docs:
                    for i in range(len(doc)):
                        if doc[i].pos_ == 'ADJ' and \
                                doc[i].text not in neg_adj and \
                                doc[i].text not in first_candidates_einzel:
                            adjectives.append((words_list[i][0], doc[i].lemma_))
                return adjectives, first_candidates

            # stop-words removal
            words_list = [item for item in words_list if item[1] not in self.stops and item[1] != 'The']

            # if a glossary term consists of a single character, drop it
            for element in words_list[:]:
                if len(element[1]) <= 1:
                    words_list.remove(element)

            # identity term to term map
            term_to_original_map = {term: term for term in [x[1] for x in words_list]}

            # link every word to its vector
            term_to_split_map, split_to_vector_map = self.word_list_to_vector(words_list,
                                                                              original_to_split=term_to_original_map)
            for i in range(len(words_list)):
                words_list[i].append(split_to_vector_map[term_to_split_map[words_list[i][1]]])

            # chunk to noun phrases
            # join per id to detect noun phrases
            # noun-chunks parsing not relevant if only verbs or process verbs  are to be detected
            if get_verbs or on_process_verbs:
                return words_list, first_candidates

            return words_list, first_candidates

        def list_of_lists_to_sentences(self, list_):
            """
            Gets list of (id, word) and converts it to a list of sentences.
            """
            sentences = [''] * len(self.reqs)
            for id_ in range(len(self.reqs)):
                for x in list_:
                    if x[0] == id_ + 1:
                        sentences[id_] += x[1] + ' '
            return sentences

        def word_list_to_vector(self, words_list, original_to_split=None):
            """
            Takes a words list as per the pipeline
            and returns a map to its co-occurence vector representation
            """

            # split according to clustering
            # if the result of recursive splitting is empty (''), return x[0] as it is
            split_words_list = []
            for x in words_list:
                try:  # KeyError entails that the word in question is a trigram and is
                    # not in he original_to_split map
                    split_word = self.recursive_split(original_to_split[x[1]])
                except KeyError:
                    split_word = self.recursive_split(x[1])
                if split_word != '':
                    split_words_list.append([x[0], split_word])
                else:
                    split_words_list.append([x[0], x[1]])

            # lemmatize
            lemma = []
            for i in range(len(split_words_list)):
                # filter out non words
                # r'[^\W\d]+-?[^\W\d]+
                split_words_list[i][1] = ' '.join(re.findall(r'[^\W\d]+', split_words_list[i][1]))
                doc2 = self.nlp(split_words_list[i][1])
                for token in doc2:
                    lemma += [token.lemma_]
                split_words_list[i][1] = ' '.join(lemma)
                lemma = []

            # map original form to split form
            term_to_split_map = {term: split_form for term, split_form in zip([x[1] for x in words_list],
                                                                              [x[1] for x in split_words_list])}
            # replace '' in split values with the origin word
            for item in term_to_split_map.keys():
                if term_to_split_map[item] == '':
                    term_to_split_map[item] = item
            # reconstruct sentences from split form
            sentences = self.list_of_lists_to_sentences(split_words_list)
            # transform document to co-occurrence matrix (from split forms' sentences)
            self.co_occurrence_object.transform_document(sentences)
            # map term to vector
            # adding term to split map for distance computation during the clustering
            split_to_vector_map = {}
            for i in range(len(words_list)):
                split_to_vector_map[
                    term_to_split_map[words_list[i][1]]] = self.co_occurrence_object.get_vector_from_term(
                    term_to_split_map[words_list[i][1]])
            return term_to_split_map, split_to_vector_map

        def log_likelihood(self, word, vorkommen):

            # Afokorpus is a fixed dataset
            n_Afokorpus = 76862
            try:
                w_Afokorpus = self.statistics[word][1]
            except KeyError:
                w_Afokorpus = 0.00001

            # wordfreq is a fixed dataset
            n_wordfreq = 569010
            w_wordfreq = vorkommen[word][2]
            # smooth 0
            if w_wordfreq == 0:
                w_wordfreq = 0.00001

            # Dataset
            w_dataset = vorkommen[word][1]
            n_dataset = self.total_number_of_words

            coeff_Afokorpus = (w_Afokorpus + w_dataset) / (n_Afokorpus + n_dataset)
            coeff_wordfreq = (w_wordfreq + w_dataset) / (n_wordfreq + n_dataset)

            E_Afokorpus = n_Afokorpus * coeff_Afokorpus
            E_wordfreq = n_wordfreq * coeff_wordfreq

            return [2 * (w_dataset + log(w_dataset / (n_dataset * coeff_Afokorpus)) + w_Afokorpus * log(
                w_Afokorpus / E_Afokorpus)),
                    2 * (w_dataset + log(w_dataset / n_dataset * coeff_wordfreq) + w_wordfreq *
                         log(w_wordfreq / E_wordfreq))]

        def get_output(self, final, maps=None, word_class=None, first_candidates=None, out=None,
                       cluster=None):
            """



            :param map_:
            :param on_abbreviations:
            :param word_class:
            :param final: the final tuples list of the pipeline
            :param first_candidates: whether to take into account abbreviations
            :param out: defines an output file
            :param gold: define the csv file with gold standard glossary terms
            :param cluster: True if we choose to perform clustering on output
            :return: output dataframe after being stored in filesystem as csv
            """
            # Get the output
            _out, sentences = self.occurence_to_csv(final, maps=None, word_class=word_class)

            # --
            ## append the abbvs to the output and sort
            # --
            if first_candidates:
                abbvs, sentences = self.occurence_to_csv(first_candidates, word_class=word_class)
                _out = _out.append(abbvs)

            # Use _out (dataframe) data and cluster

            if cluster:
                # only run clustering for non empty glossary lists
                if len(_out) >= 2:
                    _out = create_clusters(_out, maps, self.lang, sentences)
            else:
                _out['Cluster_Ergebnis'] = ['N/A'] * len(_out)
            _out. \
                sort_values(by=['absolute_term_frequency_in_requirements_set'], ascending=False). \
                to_csv(out, columns=[
                'term',
                'word_class',
                'alphabetical_order',
                'relative_term_frequency_in_requirements_set',
                'relative_term_frequency_in_Allgemeinkorpus',
                'relative_term_frequency_in_GroßerAfoTopf',
                'log-likelihood-based-on-Afokorpus',
                'log-likelihood-based-on-wordfreq',
                'ratio_of_relative_term_frequencies_to_GroßerAfoTopf',
                'requirements_id_where_term_appears'
            ])
            return _out

        def occurence_to_csv(self, _list, maps=None, word_class=None, out=None):
            """
            Translates a corpus of words to uniques, and outputs to csv
            Input: a list of tuples of (id, word)
            """

            # dictionary of occurence/importance of words
            vorkommen = {}

            # individual words post-pipeline
            list_words = [item[1] for item in _list]
            # their number
            # total_number_of_words = len(list_words)

            uniques = list(set(list_words))
            for unique in uniques:
                increment = 0
                for x in range(len(_list)):
                    if unique == _list[x][1]:
                        increment += 1
                    vorkommen[unique] = (increment,)

            # Frequency
            if self.lang == 'EN':
                for unique in uniques:
                    vorkommen[unique] += (vorkommen[unique][0] / self.total_number_of_words,
                                          word_frequency(unique, 'en'))
            else:
                for unique in uniques:
                    vorkommen[unique] += (vorkommen[unique][0] / self.total_number_of_words,
                                          word_frequency(unique, 'de'))

            # ids
            ids = []
            temp = []
            for word in vorkommen.keys():
                for tuple_ in _list:
                    if word == tuple_[1] and tuple_[0] not in temp:
                        temp += [tuple_[0]]
                ids.append(temp)
                temp = []

            # vorkommen
            occurrence = [item[0] for item in list(vorkommen.values())]
            # häufigkeit_anforderungen
            h_a = [item[1] for item in list(vorkommen.values())]
            # häufigkeit allgm
            h_allg = [float(str(item[2])) for item in list(vorkommen.values())]

            test = []
            for x in range(len(h_a)):
                if h_a[x] > h_allg[x]:
                    test += [1]
                else:
                    test += [0]

            # read json statistics and parse to h_afokorpus
            h_afokorpus = []

            for word in list(vorkommen.keys()):
                try:
                    h_afokorpus.append(self.statistics[word][1])
                except KeyError:
                    h_afokorpus.append(0)

            word_class = [self.nlp(x)[0].pos_ if word_class is None else word_class for x in list(vorkommen.keys())]

            # loglikelihood list
            log_likelihood = [self.log_likelihood(x, vorkommen) for x in list(vorkommen.keys())]

            # to pandas
            columns = {
                'term': list(vorkommen.keys()),
                'word_class': word_class,
                'alphabetical_order': [''] * len(word_class),
                'absolute_term_frequency_in_requirements_set': occurrence,
                'relative_term_frequency_in_requirements_set': h_a,
                'relative_term_frequency_in_Allgemeinkorpus': h_allg,
                'relative_term_frequency_in_GroßerAfoTopf': h_afokorpus,
                'log-likelihood-based-on-Afokorpus': [x[0] for x in log_likelihood],
                'log-likelihood-based-on-wordfreq': [x[1] for x in log_likelihood],
                'ratio_of_relative_term_frequencies_to_GroßerAfoTopf': [round(a / b, 3) if b != 0 else 0 for a, b in
                                                                        zip(h_a, h_afokorpus)],
                'requirements_id_where_term_appears': ids
            }

            df = pd.DataFrame(columns,
                              index=range(len(vorkommen)))

            # tf-idf
            # reconstruction of documents (sentences) is needed
            # if empty _list, skip
            if _list:
                sentences = self.list_of_lists_to_sentences(_list)
                tfidf_dict = tfidf_encode(sentences, list(vorkommen.keys()))
                tf_idf_list = list(tfidf_dict.values())
                df.insert(loc=10, column="TF-IDF", value=tf_idf_list)
            else:
                sentences = None
                df.insert(loc=10, column="TF-IDF", value=[])
            # read abbreviations definition
            abbrvs_def = json.load(open("GTETE_backend/data/DBAbkBw.json", 'r'))

            abkrz_ausgesch = []
            for word in list(vorkommen.keys()):
                try:
                    abkrz_ausgesch.append(abbrvs_def[word])
                except KeyError:
                    abkrz_ausgesch.append('')
            df.insert(loc=1, column="definition from DBAbkBw", value=abkrz_ausgesch)
            # else:
            definition_term = []
            with open("GTETE_backend/data/DBTermBw.json", 'r') as f_:
                definitions = json.load(f_)
            for word in list(vorkommen.keys()):
                try:
                    definition_term.append(definitions[word])
                except KeyError:
                    definition_term.append('')
            df.insert(loc=2, column="definition from DBTermBw", value=definition_term)
            df.sort_values(by=['absolute_term_frequency_in_requirements_set'], inplace=True, ascending=False)
            df.reset_index(inplace=True, drop=True)
            if out:
                df.to_csv(out)
            # create a split to vector map with the order in the sorted dataframe
            split_to_vector_map = {split_term: vector for split_term, vector in zip()}
            return df, sentences

    def df_to_dicts(df):
        return [
            {a: b for a, b in zip(df.columns, df.values.tolist()[i])} for i in range(len(df))
        ]

    class ChunksPipeline(Pipeline):
        def __init__(self,
                     hunspell_str,
                     spacy_str,
                     req_dataset,
                     co_occurrence_object,
                     lang=None,
                     csv_template=None,
                     gold=None):
            self.gold = gold
            super().__init__(hunspell_str,
                             spacy_str,
                             req_dataset,
                             co_occurrence_object,
                             lang=lang,
                             csv_template=csv_template)

            if self.lang == 'DE':
                # - new exception stop words, only relevant to noun-chunks pipeline
                exception_stops = [item[:-1] for item in
                                   open('GTETE_backend/models/exceptionStopWords.txt', 'r').readlines()]
                self.stops = [item for item in self.stops if item not in exception_stops]

        def run_pipeline_chunks(self):
            req_chunks_tuple = []
            id_ = 1
            tmp = ''
            original_to_split_map = {}
            for doc in self.docs:
                # req = ' '.join(re.findall(r'\w+', req))
                tmp = None
                for chunk in doc.noun_chunks:
                    if self.lang == 'EN':
                        # remove 'the ' or 'The ' or 'a '
                        custom_stop_words = [
                            'the ',
                            'The ',
                            'a '
                        ]
                        for s in custom_stop_words:
                            if chunk.text.startswith(s):
                                tmp = chunk.lemma_.replace(s, '')
                                req_chunks_tuple += [[id_, tmp]]
                    elif self.lang == 'DE':
                        custom_stop_words = [
                            'der ',
                            'Der ',
                            'die ',
                            'Die ',
                            'Das ',
                            'das '
                        ]
                        # append adjusted spacy stop-words
                        custom_stop_words += self.stops
                        if chunk.text in custom_stop_words:
                            continue
                        for s in custom_stop_words:
                            if chunk.text.split()[0] == s[:-1]:
                                tmp = chunk.text.replace(s[:-1] + ' ', '')
                                # get lemmatized noun chunks, without the '-'
                                words = ' '.join(re.findall(r'\w+', tmp))
                                lemmatized_words = ' '.join(
                                    [x.lemma_ if x.pos_ != 'DET' else x.text for x in self.nlp(words)])
                                original_to_split_map[tmp] = lemmatized_words
                                # remove parentheses
                                req_chunks_tuple += [[id_, re.sub(r'[()]', '', tmp)]]
                                break

                    if not tmp:
                        words = ' '.join(re.findall(r'\w+', chunk.text))
                        lemmatized_words = ' '.join([x.lemma_ if x.pos_ != 'DET' else x.text for x in self.nlp(words)])
                        original_to_split_map[chunk.text] = lemmatized_words
                        req_chunks_tuple += [[id_, re.sub(r'[()]', '', chunk.text)]]

                    tmp = None

                trigrams = []
                text_and_pos_tuples = [[token.text, token.pos_] for token in doc]
                triGrams = [text_and_pos_tuples[i:i + 3] for i in range(len(text_and_pos_tuples) - 2)]
                # Some pattern intelligence is needed
                # The first part of the pattern can be lemmatized.
                # The article and the third part should be correctly lemmatized, according to german
                # grammatic rules.
                for tri_gram in triGrams:

                    if ((tri_gram[0][1] == "NOUN")
                            and (tri_gram[1][1] == "DET"
                                 or tri_gram[1][1] == "ADP")
                            and (tri_gram[2][1] == "NOUN")):
                        trigrams.append(
                            [id_, ' '.join([tri_gram[i][0] for i in range(len(tri_gram))])])
                req_chunks_tuple += trigrams
                id_ += 1
            # remove '' from req_chunks_tuple
            req_chunks_tuple = [x for x in req_chunks_tuple if x[1] != '']
            words_list_to_vector_map, term_to_split_map = self.word_list_to_vector(req_chunks_tuple,
                                                                                   original_to_split=original_to_split_map)
            '''# remove - between terms
            for i in range(len(req_chunks_tuple)):
                req_chunks_tuple[i][1] = ' '.join(re.findall(r'\w+', req_chunks_tuple[i][1]))
                # append vector'''

            # ship term to vector map
            return req_chunks_tuple, words_list_to_vector_map, term_to_split_map

    class Main:
        def __init__(self, dataset, template=None, output=None):
            self.dataset = dataset
            self.template = template
            self.output = output
            # get the first sentence
            first_sentence = self.dataset[0]['text']
            self.language = detectLanguage(first_sentence)
            if self.language != 'EN' and self.language != 'DE':
                raise Exception('Input error: language can only be \'EN\' or \'DE\'')
            self.folder = 'GTETE_backend/csv_output/'
            self.co_occurrence = CoOccurrence()
            if self.language == 'DE':
                self.folder += 'German/'
                self.pipeline = Pipeline('de_DE',
                                         'de_core_news_sm',
                                         self.dataset,
                                         self.co_occurrence,
                                         lang=self.language,
                                         csv_template=self.template)

            elif self.language == 'EN':
                self.folder += 'English/'
                self.pipeline = Pipeline('en_US',
                                         'en_core_web_sm',
                                         # 'data/merged.csv',
                                         self.dataset,  # 'EN_reqs.csv',
                                         self.co_occurrence,
                                         lang=self.language,
                                         csv_template=self.template)

        def call_tokens_pipeline(self,
                                 word_class=None,
                                 on_abbreviations=None,
                                 get_verbs=None,
                                 on_process_verbs=None,
                                 get_adjectives=None,
                                 cluster=None):

            # The pipeline is called from Separated/Merged twice:
            # 1. to get tokens without abbreviations (final), and
            # 2. to get abbreviations only (abbreviaitons).
            # This behaviour can be avoided so that the running time is improved.

            final, abbreviations = self.pipeline.run_pipeline_tokens(get_verbs=get_verbs,
                                                                     on_process_verbs=on_process_verbs,
                                                                     get_adjectives=get_adjectives)

            if on_abbreviations:
                # run the pipeline on abbrevations only
                # check with negative list on abbreviations here
                tokens_df = self.pipeline.get_output(abbreviations, word_class=word_class, cluster=cluster)
            else:
                # run the pipeline on tokens only (in separated branch)
                tokens_df = self.pipeline.get_output(final, word_class=word_class, cluster=cluster)
            return tokens_df

        def call_chunks_pipeline(self, word_class=None, cluster=None):
            if self.language == 'DE':
                chunks_pipeline = ChunksPipeline('de_DE',
                                                 'de_core_news_sm',
                                                 self.dataset,
                                                 # append ../ before for running here
                                                 self.co_occurrence,
                                                 lang='DE',
                                                 csv_template=self.template
                                                 )
            elif self.language == 'EN':
                chunks_pipeline = ChunksPipeline('en_US',
                                                 'en_core_web_sm',
                                                 self.dataset,  # '../../../Schreibtisch/EN_reqs.csv',
                                                 self.co_occurrence,
                                                 lang=self.language,
                                                 csv_template=self.template
                                                 )
            occurrence, *maps = chunks_pipeline.run_pipeline_chunks()
            chunks_df = chunks_pipeline.get_output(occurrence, maps=maps, word_class=word_class, cluster=cluster)
            return chunks_df

        def print_log(self, string_):
            folder_string = self.folder + string_ + 'log.txt'
            with open(folder_string, 'w') as f:
                f.write('''
    Latest execution for: 
        dataset: %s
        language: %s
        template conform: %s 
                ''' % (self.dataset, self.language, str(self.template)))

    relevantWordsEN = {'the', 'must', 'shall', 'should', 'will', 'have', 'user', 'allow', 'provide', 'ability', 'be',
                       'able', 'to', 'if', 'with', 'as', 'of', 'and', 'are', 'not', 'any', 'a', 'that', 'it', 'for',
                       'not',
                       'on', 'with', 'do', 'at', 'this', 'but', 'by', 'from', 'or', 'one', 'all', 'would', 'there',
                       'what',
                       'up', 'out', 'about', 'who', 'get', 'which', 'go', 'when', 'can', 'like', 'time', 'no', 'just',
                       'take', 'into', 'some', 'other', 'than', 'only', 'than', 'its', 'over', 'new', 'these'}

    relevantWordsDE = {'der', 'auch', 'die', 'das', 'muss', 'kann', 'soll', 'wird', 'haben', 'Nutzer', 'Bediener',
                       'Möglichkeit', 'stellen', 'zu', 'fähig', 'sein', 'ist', 'falls', 'solange', 'mit', 'auf', 'eine',
                       'den', 'von', 'sich', 'des', 'für', 'im', 'dem', 'nicht', 'ein', 'Die', 'als', 'auch', 'es',
                       'werden', 'aus', 'hat', 'nach', 'bei', 'einer', 'Der', 'um', 'am', 'sind', 'noch', 'wie',
                       'einem',
                       'über', 'Das', 'so', 'zum', 'und', 'oder', 'vor', 'zur', 'bis', 'mehr', 'durch', 'Prozent',
                       'gegen',
                       'vom', 'wenn', 'unter', 'zwei', 'zwischen'}

    def detectLanguage(sentence):
        # language detector
        splittedSentence = set(sentence.split())

        CountEN = len(splittedSentence.intersection(relevantWordsEN))

        CountDE = len(splittedSentence.intersection(relevantWordsDE))

        if CountEN > CountDE:
            print(str(CountEN) + ', ' + str(CountDE))

            return 'EN'

        if CountDE > CountEN:
            return 'DE'

    class Merged(Main):
        """
        Generates the merged outputs
        of words and noun-chunks pipelines
        """

        def merged_outputs(self):

            directory = self.folder + 'merged/'
            if not os.path.exists(directory):
                os.makedirs(directory)

            dfs = [
                self.call_tokens_pipeline(),  # tokens without abbvs
                # transform to a json response
                self.call_tokens_pipeline('Abbreviation', on_abbreviations=True),  # abbrviations
                # run clustering on NC pipeline only!
                self.call_tokens_pipeline('VERB', get_verbs=True),  # verbs
                self.call_tokens_pipeline('VERB', get_verbs=True, on_process_verbs=True),  # process verbs only
                self.call_tokens_pipeline('ADJ', get_adjectives=True)  # adjectives
            ]

            # merge dfs
            # check for duplicates ---

            tokens_df = pd.concat(dfs)
            chunks_df = self.call_chunks_pipeline('Noun Chunk', cluster=True)  # chunks
            # compare chunks_df to tokens_df
            # if item in tokens_df in chunks_df : remove it

            for x in chunks_df['term']:
                for y in tokens_df['term']:
                    if y in x:
                        # drop y
                        tokens_df = tokens_df[tokens_df['term'] != y]
            # remove split_terms from chunks_df
            chunks_df = chunks_df.drop(columns=['split_form'])
            # merge token_df with chunks_df
            full_df = tokens_df.append(chunks_df). \
                sort_values(by=['absolute_term_frequency_in_requirements_set'], ascending=False)

            # alphabetical ordering goes here
            # extract terms from the df
            terms = full_df['term']
            alpha_order_map = {term: order for order, term in enumerate(sorted(terms))}
            full_df['alphabetical_order'] = [alpha_order_map[term] for term in terms]

            full_df = full_df.reset_index(drop=True)
            full_df = full_df.fillna('N/A')
            # remove duplicate entries
            # store duplicate indexes
            repeated_indexes = []
            for i in range(len(full_df)):
                for j in range(i + 1, len(full_df)):
                    if full_df.at[i, 'term'] == full_df.at[j, 'term'] and \
                            i not in repeated_indexes:
                        repeated_indexes.append(i)
            for u in repeated_indexes:
                full_df = full_df.drop(u)
            full_df = full_df.reset_index(drop=True)
            # to file for exploration
            full_df.to_csv(directory + 'merged.csv')
            self.print_log('merged/')
            return df_to_dicts(full_df.sort_values(by='absolute_term_frequency_in_requirements_set', ascending=False))

    merged = Merged(requirements_list, template=False)  # False for test dataset

    if connexion.request.is_json:
        # requirements_list = [SoftwareRequirement.from_dict(d) for d in connexion.request.get_json()]  # noqa: E501
        return merged.merged_outputs()


def ping_get():  # noqa: E501
    """ping_get

    Tests whether the server is in a responding state or not. No underlying logic involved. # noqa: E501


    :rtype: str
    """
    return 'pong'


def version_get():  # noqa: E501
    """version_get

    Returns the backend version number. # noqa: E501


    :rtype: str
    """
    return '1.0-SNAPSHOT'
