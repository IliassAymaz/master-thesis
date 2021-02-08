import json
import os
from charsplit import Splitter
import spacy
import nltk
import re
import pandas
import pprint
import pickle
import time
from textblob_de import TextBlobDE as TextBlob
# from germalemma import GermaLemma

# os.environ['TREETAGGER_HOME'] = 'TreeTagger/'
from treetagger import TreeTaggerChunker, TreeTagger
import hunspell

spellchecker = hunspell.HunSpell('/usr/share/hunspell/{}.dic'.format('de_DE'),
                                 '/usr/share/hunspell/{}.aff'.format('de_DE'))

# lemmatizer = GermaLemma()
tt_nc = TreeTaggerChunker(path_to_treetagger='TreeTagger/', language='german')
tt = TreeTagger(path_to_treetagger='TreeTagger', language='german')

nltk.data.path.append('nltk_data/')
splitter = Splitter()
pp = pprint.PrettyPrinter()


def recursive_split(glossary_term, nlp) -> str:
    ergebnis = []
    for word in glossary_term.split():
        if len(word) < 14:
            # check if it ends with s and remove it
            # if, after removing s, term is tagged as NOUN, then remove s
            # and return it as it is
            if word[-1] == 's' and len(word) > 1:
                if spellchecker.spell(word[:-1]):
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
                ergebnis += [item[:-1] if item[-1] == 's' and nlp(item)[0].pos_ != 'PROPN' and spellchecker.spell(
                    item[:-1]) else item for
                             item in
                             splitter.split_compound(word)[0][1:]]
    output = [glossary_term, ' '.join(ergebnis)]
    if output[1] == glossary_term:
        return glossary_term
    O = []
    for x in output[1].split():
        O.append(recursive_split(x, nlp))
    return ' '.join(O)


class Sentence:
    """
    Models one sentence with its desired attributes.
    """

    def __init__(self):
        self._sentence = None
        self._language = None
        self._lemmatized_spaCy = []
        self._lemmatized_NLTK = []
        self._lemmatized_germalemma = []
        self._lemmatized_tree_tagger = []
        self._pos_tags_spaCy = []
        self._pos_tags_NLTK = []
        self._pos_tags_NLTK_2 = []
        self._pos_tags_tree_tagger = []
        self._noun_chunks = []
        self._noun_chunks_NLTK = []
        self._noun_chunks_tree_tagger = []
        self._nlp = None
        self._doc = None
        self._blob = None

    @property
    def sentence(self):
        return self._sentence

    @sentence.setter
    def sentence(self, sentence):
        self._sentence = sentence

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, lang):
        self._language = lang

    @property
    def nlp(self):
        return self._nlp

    @nlp.setter
    def nlp(self, nlp_):
        self._nlp = nlp_

    @property
    def doc(self):
        self._doc = self.nlp(self._sentence)
        return self._doc

    @property
    def blob(self):
        self._blob = TextBlob(self._sentence)
        return self._blob

    @property
    def lemmatized_spaCy(self):
        seen = []
        for token in self.doc:
            if token.pos_ == 'ADJ' or token.pos_ == 'VERB':
                if (token.text, token.lemma_) in seen:
                    continue
                else:
                    seen.append((token.text, token.lemma_))
                    self._lemmatized_spaCy.append((token.text, token.lemma_))
        # add lemmatized noun chunk
        tuples = []
        for chunk in self.doc.noun_chunks:
            if chunk[0].is_stop:
                tuples += [
                    (' '.join(re.findall(r'[\w-]+', ' '.join(chunk.text.split()[1:]))),
                     ' '.join(re.findall(r'[\w-]+', ' '.join(chunk.lemma_.split()[1:])))
                     )
                ]

                # chunks.append(' '.join(re.findall(r'[\w-]+', x)) for x in chunk.text.split()[1:])
            else:
                tuples += [
                    (' '.join(re.findall(r'[\w-]+', chunk.text)),
                     ' '.join(re.findall(r'[\w-]+', chunk.lemma_))
                     )
                ]
            # remove empties and duplicates from found chunks

        self._lemmatized_spaCy += [x for x in tuples if x[0]]
        return self._lemmatized_spaCy

    @property
    def lemmatized_NLTK(self):
        seen = []
        blob = self.blob
        for word, lemma, tag in zip(blob.words, blob.words.lemmatize(), blob.tags):
            if tag[1] == 'JJ' or tag[1] == 'VB':
                if (word, lemma) in seen:
                    continue
                else:
                    seen.append((word, lemma))
                    self._lemmatized_NLTK.append((word, lemma))
        tmp = []
        for chunk in self.blob.noun_phrases:
            # blob the chunk and lemmatize it
            lemmatized_chunk = ' '.join(TextBlob(chunk).words.lemmatize())
            tmp.append((chunk, lemmatized_chunk))
        # self._lemmatized_NLTK.append(self.noun_chunks_NLTK)
        self._lemmatized_NLTK += tmp
        return self._lemmatized_NLTK

    @property
    def lemmatized_germalemma(self):
        # to be replaced to make sense of the usage of a single whole tool
        # as a third option: TreeTagger
        seen = []
        for word in self.doc:
            if word.pos_ == 'ADJ' or word.pos_ == 'VERB':
                lemma = lemmatizer.find_lemma(word.text, word.pos_)
                if (word, lemma) in seen:
                    continue
                else:
                    seen.append((word, lemma))
                    self._lemmatized_germalemma.append((word, lemma))
        return self._lemmatized_germalemma

    @property
    def lemmatized_tree_tagger(self):
        seen = []
        for element in tt.tag(self._sentence):
            if element[1].startswith('ADJ') or element[1].startswith('V'):
                if (element[0], element[2]) in seen:
                    continue
                else:
                    seen.append((element[0], element[2]))
                    self._lemmatized_tree_tagger.append((element[0],
                                                         element[2] if element[2] != '<unknown>' else element[0]))

        tmp = []
        for item in self.noun_chunks_tree_tagger:
            tmp.append((item[0],
                        ' '.join([x[2] if x[2] != '<unknown>' else x[0] for x in tt.tag(item[0])])))
        self._lemmatized_tree_tagger += tmp
        return self._lemmatized_tree_tagger

    @property
    def pos_tags_spaCy(self):
        # consider adjectives and verbs
        seen = []
        for token in self.doc:
            if token.pos_ == 'ADJ' or token.pos_ == 'VERB':
                if (token.text, token.pos_) in seen:
                    continue
                else:
                    seen.append((token.text, token.pos_))
                    self._pos_tags_spaCy.append((token.text, token.pos_))
        return self._pos_tags_spaCy

    @property
    def pos_tags_NLTK(self):
        tokens = nltk.word_tokenize(self._sentence, language='german')
        seen = []
        tags = tagger.tag(re.findall(r'[\w-]+', self._sentence))
        # keep only ADJ* and V*
        tags = [x for x in tags if x[1].startswith('ADJ') or x[1].startswith('V')]
        for pos in tags:
            if pos in seen:
                continue
            else:
                seen.append(pos)
                self._pos_tags_NLTK.append(pos)
        return self._pos_tags_NLTK

    @property
    def pos_tags_NLTK_2(self):
        # blob
        blob = self._blob
        seen = []
        for word, tag in zip(blob.words, blob.tags):
            if tag[1] == 'JJ' or tag[1] == 'VB':
                if (word, tag) in seen:
                    continue
                else:
                    seen.append((word, tag[1]))
                    self._pos_tags_NLTK_2.append((word, tag[1]))
        return self._pos_tags_NLTK_2

    @property
    def pos_tags_tree_tagger(self):
        seen = []
        for element in tt.tag(self._sentence):
            if element[1].startswith('V'):
                tmp = (element[2] if element[2] != '<unknown>' else element[0], 'V')
            if element[1].startswith('ADJ'):
                tmp = (element[2] if element[2] != '<unknown>' else element[0], 'ADJ')
                seen.append(tmp)
                self._pos_tags_tree_tagger.append(tmp)
        return self._pos_tags_tree_tagger

    @property
    def noun_chunks(self):
        # consider noun chunks
        found_chunks = []
        for chunk in self.doc.noun_chunks:
            if chunk[0].is_stop:
                found_chunks.append(' '.join(re.findall(r'[\w-]+', ' '.join(chunk.text.split()[1:]))))
                # found_chunks.append(' '.join(re.findall(r'[\w-]+', x)) for x in chunk.text.split()[1:])
            else:
                found_chunks.append(' '.join(re.findall(r'[\w-]+', chunk.text)))
        # remove empties and duplicates from found chunks
        found_chunks = [x for x in found_chunks if x]
        found_chunks = list(set(found_chunks))
        # split and lemmatize last part

        split_form = []
        lemmatized_split_form = []
        for chunk in found_chunks:
            split_form.append(recursive_split(chunk, self._nlp))
        for element in split_form:
            # lemmatize last part
            if element:
                lemmatized_split_form.append(' '.join([' '.join(element.split()[:-1]),
                                                       self._nlp(element.split()[-1])[0].lemma_])
                                             )
        # return (chunk, lemmatized_spaCy split form)
        self._noun_chunks = [(chunk, split) for chunk, split in zip(found_chunks, lemmatized_split_form)]
        return self._noun_chunks

    @property
    def noun_chunks_NLTK(self):
        # blob
        blob = self.blob
        seen = []
        for chunk in blob.noun_phrases:
            split_chunk = ' '.join([recursive_split(x, self._nlp) for x in chunk.split()])
            split_doc = self._nlp(split_chunk)
            tmp = []
            for token in split_doc:
                tmp.append(token.lemma_)
            lemmatized_chunk = ' '.join(tmp)
            if (chunk, lemmatized_chunk) in seen:
                continue
            else:
                seen.append((chunk, lemmatized_chunk))
                self._noun_chunks_NLTK.append((chunk, lemmatized_chunk))

        return self._noun_chunks_NLTK

    @property
    def noun_chunks_tree_tagger(self):
        found_chunks = []
        tmp = []
        in_nc = False
        lemmatized_chunks = []
        for element in tt_nc.parse(self._sentence):
            if element == ['<NC>']:
                in_nc = True
                continue
            if element == ['</NC>']:
                in_nc = False
                found_chunks.append(' '.join(tmp))
                tmp = []
                continue
            if in_nc:
                # exclude articles
                if element[1] != 'ART' and element[1] != 'PRELS':
                    tmp.append(element[0])
        # clean empties from found_chunks
        found_chunks = [x for x in found_chunks if x != '']
        for chunk in found_chunks:
            split_chunk = ' '.join([recursive_split(x, self._nlp) for x in chunk.split()])
            lemmatized_chunks.append(' '.join([x[2] if x[2] != '<unknown>' else x[0] for x in tt.tag(split_chunk)]))
        for chunk, lemmatized_chunk in zip(found_chunks, lemmatized_chunks):
            self._noun_chunks_tree_tagger.append((chunk, lemmatized_chunk))

        return self._noun_chunks_tree_tagger


class NLPComparator:
    """
    Populates output with Sentence instances based on a json dataset.
    """

    def __init__(self):
        self._language = None
        self._nlp = None
        self._dataset = None
        self._output = []

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, lang):
        self._language = lang

    @property
    def nlp(self):
        if self._language == 'DE':
            self._nlp = spacy.load('de_core_news_sm')
        elif self._language == 'EN':
            self._nlp = spacy.load('en_core_news_sm')
        return self._nlp

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, data):
        # data is json
        self._dataset = [item['text'] for item in json.load(data)]

    # add the 'comparison' feature
    # lemmatized_spaCy
    # lemmatized_NLTK
    # lemmatized_gensim

    @property
    def output(self):
        pg_bar = [
            ['|  ', '  ', '  ', '  ', '  ', '  |'],
            ['|██', '█ ', '  ', '  ', '  ', '  |'],
            ['|██', '██', '██', '  ', '  ', '  |'],
            ['|██', '██', '██', '██', '█ ', '  |'],
            ['|██', '██', '██', '██', '██', '██|']]

        def progress_bar(t, len_dataset):

            if float(t / len_dataset) < 0.25:
                print(">> Comparison in progress ... %s %.1f %%" % (''.join(pg_bar[0]), float(t / len_dataset * 100)),
                      end='\r')
            elif 0.25 <= float(t / len_dataset) < 0.5:
                print(">> Comparison in progress ... %s %.1f %%" % (''.join(pg_bar[1]), float(t / len_dataset * 100)),
                      end='\r')
            elif 0.5 <= float(t / len_dataset) < 0.75:
                print(">> Comparison in progress ... %s %.1f %%" % (''.join(pg_bar[2]), float(t / len_dataset * 100)),
                      end='\r')
            elif 0.75 <= float(t / len_dataset) < 1:
                print(">> Comparison in progress ... %s %.1f %%" % (''.join(pg_bar[3]), float(t / len_dataset * 100)),
                      end='\r')
            elif float(t / len_dataset) == 1:
                print(">> Comparison in progress ... %s %.1f %%" % (''.join(pg_bar[4]), float(100)))
                time.sleep(1)
                print(">> Done.")

        execution_time_spacy, execution_time_nltk, execution_time_germalemma = 0, 0, 0

        # call Sentence
        t = 1
        time_tracker = {'spaCy Lemmatization': 0,
                        'spaCy POS-Tagging': 0,
                        'spaCy Noun Chunking': 0,
                        'NLTK Lemmatization': 0,
                        'NLTK POS-Tagging': 0,
                        'NLTK Noun Chunking': 0,
                        'TreeTagger Lemmatization': 0,
                        'TreeTagger POS-Tagging': 0,
                        'TreeTagger Noun Chunking': 0
                        }

        for phrase in self.dataset:
            progress_bar(t, len(self._dataset))
            output_object = {}

            sentence = Sentence()
            sentence.language = self._language
            sentence.nlp = self._nlp
            sentence.sentence = phrase
            output_object['sentence'] = sentence.sentence

            start = time.time()
            output_object['lemmatized_spaCy'] = '\n'.join([str(x) for x in sentence.lemmatized_spaCy])
            time_tracker['spaCy Lemmatization'] += time.time() - start

            start = time.time()
            output_object['pos tags spaCy'] = '\n'.join([str(x) for x in sentence.pos_tags_spaCy])
            time_tracker['spaCy POS-Tagging'] += time.time() - start

            start = time.time()
            output_object['noun chunks spaCy'] = '\n'.join([str(x) for x in sentence.noun_chunks])
            time_tracker['spaCy Noun Chunking'] += time.time() - start

            start = time.time()
            output_object['lemmatized_NLTK'] = '\n'.join([str(x) for x in sentence.lemmatized_NLTK])
            time_tracker['NLTK Lemmatization'] += time.time() - start

            start = time.time()
            output_object['pos tags NLTK'] = '\n'.join([str(x) for x in sentence.pos_tags_NLTK_2])
            time_tracker['NLTK POS-Tagging'] += time.time() - start

            start = time.time()
            output_object['noun chunks NLTK'] = '\n'.join([str(x) for x in sentence.noun_chunks_NLTK])
            time_tracker['NLTK Noun Chunking'] += time.time() - start

            start = time.time()
            output_object['lemmatized_TreeTagger'] = '\n'.join([str(x) for x in sentence.lemmatized_tree_tagger])
            time_tracker['TreeTagger Lemmatization'] += time.time() - start

            start = time.time()
            output_object['pos tags TreeTagger'] = '\n'.join([str(x) for x in sentence.pos_tags_tree_tagger])
            time_tracker['TreeTagger POS-Tagging'] += time.time() - start

            start = time.time()
            output_object['noun chunks TreeTagger'] = '\n'.join([str(x) for x in sentence.noun_chunks_tree_tagger])
            time_tracker['TreeTagger Noun Chunking'] += time.time() - start

            # output_object['pos tags TreeTagger'] = '\n'.join([str(x) for x in sentence.pos_tags_tree_tagger])
            # output_object['lemmatized_GermaLemma'] = '\n'.join([str(x) for x in sentence.lemmatized_germalemma])
            # output_object['noun chunks TreeTagger'] = '\n'.join([str(x) for x in sentence.noun_chunks_tree_tagger])
            # execution_time_germalemma += time.time() - start
            self._output.append(output_object)
            del sentence
            t += 1
        return self._output, time_tracker


# pp.pprint(gold)


class Scorer:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def set_scores(self, candidate: set, gold: set):
        self.TP += len(candidate.intersection(gold))
        self.FP += len(candidate.difference(gold))
        self.FN += len(gold.difference(candidate))

    def score(self) -> dict:
        """
        RETURNS the accuracy (F score) for candidate and gold sets.
        """

        # we compute the F-score instead of the plain accuracy
        # the TN are not determinable, it is the set of the 'rest' of words in the sentence
        # plus, it would cause imbalance

        recall = self.TP / (self.TP + self.FP)
        precision = self.TP / (self.TP + self.FN)
        f_score = round(2 * (precision * recall) / (precision + recall), 2)
        return {'recall': round(recall, 2), 'precision': round(precision, 2), 'f_score': f_score}


def main():
    tagger = pickle.load(open('data/nltk_german_tagger.pickle', 'rb'))

    nlp_c = NLPComparator()
    nlp_c.language = 'DE'
    nlp_c.dataset = open('data/OpenReqsDE.json', 'r')
    timer_spacy_start = time.time()
    nlp_c.nlp
    execution_time_spacy = time.time() - timer_spacy_start
    # print(nlp_c.noun_chunks)
    out, times = nlp_c.output
    # pp.pprint(out)

    pandas.DataFrame(data=out).to_csv('out/generated_nlp_features.csv', sep=';')
    print('---------------------')
    # load saved gold dictionary and compare
    gold = pickle.load(open('data/gold_standard_nlp.pickle', 'rb'))
    # gold = pickle.load(open('data/gold_standard_nlp_corrected_nc.pickle', 'rb'))

    # create a gold standard for NLTK pos tags, which entails
    # to generate a matching (better than manually adding a column)

    spacy_to_nltk_pos_tags = {'ADJ': 'JJ',
                              'VERB': 'VB'}

    spacy_to_tree_tagger_pos_tags = {
        'ADJ': 'ADJ',
        'VERB': 'V'
    }

    # rename to match the local nomenclature
    gold = pandas.DataFrame(data=gold).rename(columns={'pos tags': 'pos tags spaCy'}).to_dict(orient='records')
    # add an NLTK column
    for d in gold:
        d['pos tags NLTK'] = []
        t = [eval(x) for x in d['pos tags spaCy'].split('\n')]
        d['pos tags NLTK'] = '\n'.join([str((x[0], spacy_to_nltk_pos_tags[x[1]])) for x in t])
        d['pos tags TreeTagger'] = '\n'.join([str((x[0], spacy_to_tree_tagger_pos_tags[x[1]])) for x in t])

    keys = [('lemmatized_spaCy', 'lemmatized'),
            ('pos tags spaCy', 'pos tags spaCy'),
            ('noun chunks spaCy', 'noun chunks'),
            ('lemmatized_NLTK', 'lemmatized'),
            ('pos tags NLTK', 'pos tags NLTK'),
            ('noun chunks NLTK', 'noun chunks'),
            ('lemmatized_TreeTagger', 'lemmatized'),
            ('pos tags TreeTagger', 'pos tags TreeTagger'),
            ('noun chunks TreeTagger', 'noun chunks')
            ]

    results = []
    for key in keys:
        sc = Scorer()
        for generated, should_be in zip(out, gold):
            A = set(generated[key[0]].split('\n'))
            B = set(should_be[key[1]].split('\n'))
            sc.set_scores(A, B)
        result = sc.score()
        results.append(result['f_score'])
        print('result for %s: ' % str(key[0]), result)

    pp.pprint(times)

    import matplotlib
    import matplotlib.pyplot as plt
    font = {
        'weight': 'bold',
        'size': 16}

    matplotlib.rc('font', **font)

    x = [1 - x for x in results]
    y = list(times.values())

    fig = plt.figure(figsize=(14, 14))

    for i, j, key in zip(x, y, list(times.keys())):
        plt.plot([0, i], [j, j], linestyle='--', color='r')
        plt.vlines(i, 0, j, linestyles='--', colors='r')
        plt.annotate('%s' % key, (i, j), xytext=(0, -7), textcoords='offset points',
                     va='top', ha='center')
    plt.scatter(x, y)
    plt.xlabel('1 - Accuracy')
    plt.ylabel('Running Time (seconds)')
    # plt.title('Performance Comparison between spaCy and NLTK running on requirements documents')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    fig.savefig('out/performance_comparison_between_nlp_packages.png', dpi=fig.dpi)
    plt.show()


if __name__ == '__main__':
    main()
